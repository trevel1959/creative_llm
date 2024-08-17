import torch
import json
import os
import logging
import time
import re
import shutil
import itertools
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import timedelta, datetime
from string import Template
from accelerate import Accelerator

def set_logger(level):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s\t- %(message)s', datefmt="%H:%M:%S")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

def find_model_dir(model_name):
    with open("datas/model.json", "r", encoding="UTF-8") as model_file:
        models = json.load(model_file)
        
    model_dir = models.get(model_name)
    if model_dir is None:
        logger.error(f'[ERROR] No model corresponds to {config["model_name"]}')
        return
    return model_dir

def load_model(model_name):
    hf_token = "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"
    
    model_dir = find_model_dir(model_name)
    
    accelerator = Accelerator()
    device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        token = hf_token,
        trust_remote_code=True,    # for qwen, 240731 1805KZ
        padding_side='left'
        )
        
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        token = hf_token,
        device_map="auto",
        trust_remote_code = True    # for qwen, 240731 1723KZ
    )
        
    # 패딩 토큰이 없으면 추가 for qwen, 240814 1307KZ
    if not tokenizer.pad_token:
        if "Qwen" in model_dir:
            tokenizer.pad_token ="<|endoftext|>"
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        model.resize_token_embeddings(len(tokenizer))
    
    model = accelerator.prepare(model)
    
    return tokenizer, model, device

def text_models_batch(model_config, prompt, queries, max_length=1024):
    tokenizer, model, device = model_config
    model.eval()
    
    # 입력 문자열을 배치 단위로 토크나이즈
    input_strs = [prompt.safe_substitute(user_query=query) + " Assistant: \n1." for query in queries]
    inputs = tokenizer(input_strs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # 모델에 입력을 주고 추론 수행
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.9,
            top_p=0.7
        )

    # 출력 토큰을 텍스트로 변환
    output_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return output_texts

def make_and_save_answers(config, prompt, folder_path):
    with open(config["task_type"], "r", encoding="UTF-8") as task_file:
        task_data = json.load(task_file)
    
    # divide task prompt and data here 240818 0134KZ

    example_num = min(len(task_data), config["example_num"])
    model_config = config["model_config"]
    batch_data = [(task_data[i], os.path.join(folder_path, f"Q{i:02d}.json"), i) for i in range(example_num)]
    batch_size = config["batch_size"]

    if not config["overwrite"]:
        batch_data = [data for data in batch_data if not os.path.exists(data[1])]
        
    failed_queries = []
    if batch_data:
        for i in tqdm(range(0, len(batch_data), batch_size), desc="Generating answers...", total=math.ceil(len(batch_data) / batch_size)):
            batch = batch_data[i:i + batch_size]
            batch_queries, batch_file_paths, batch_indices = zip(*batch)

            inf_results = text_models_batch(model_config, prompt, batch_queries)
            parsed_strs = [parsing_text(result) for result in inf_results]
                    
            for parsed_str, file_path, index in zip(parsed_strs, batch_file_paths, batch_indices):
                if len(parsed_str) >= config["generate_answer_num"]:
                    with open(file_path, "w", encoding="UTF-8") as output_file:
                        json.dump(parsed_str[:config["generate_answer_num"]], output_file, indent=4, ensure_ascii=False)
                else:
                    failed_queries.append(index)

    return failed_queries, len(batch_data)

def parsing_text(text): 
    start_idx = text.find("1.")
    if start_idx == -1:
        return False
    text = re.sub(r':\s*\n+', ': ', text[start_idx:])   # remove blank and newline after ':' in text
    
    lines = list(filter(lambda x: x.strip(), text.split("\n"))) # split text into lines and remove empty line
    result = []
    for line in lines:
        formatted_line = re.sub(r"^\d+\.\s*", "", line) # remove list numbering for each line
        if line != formatted_line and formatted_line:   # append it to result when there is numbering and not empty
            result.append(formatted_line)

    return result

def remove_small_files(folder_path, size_limit_bytes = 512):
    removed_query = []
    
    for filename in os.listdir(folder_path):
        if filename == "metadata.json": continue
        
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and os.path.getsize(file_path) <= size_limit_bytes:
            os.remove(file_path)
            removed_query.append(int(re.search(r'\d+', filename).group()))
            
    return removed_query

def making_instructions(config):
    prompt_format = {
        "llama2chat": Template("""<s>[INST] <<SYS>>$system_prompt$system_stimuli\n<</SYS>>\n\n$user_query [/INST]"""),  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/
        "qwenchat": Template("""<s>$system_prompt$system_stimuli\n$user_query"""),   # temp
        "mistralinst": Template("""<s>[INST] $system_prompt$system_stimuli\n# question:\n$user_query [/INST]"""),   # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/discussions/49
        "default": Template("""<s>[INST] $system_prompt$system_stimuli\n$user_query [/INST]""") # temp
    }
    system_prompt = f"""\nFor the following questions, generate {config["generate_answer_num"]} CREATIVE and ORIGINAL ideas with detailed explanations."""
    system_stimuli = config["stimuli"]["text"]
    
    prompt_temp = prompt_format.get(config["model_name"], prompt_format["default"]).safe_substitute(
        system_prompt = system_prompt,
        system_stimuli = system_stimuli
    )

    return Template(prompt_temp)

def process_task(config):
    logger = set_logger(logging.INFO)
    
    logger.info("Task Start.")
    logger.info({k: v for k, v in config.items() if k != "model_config"})
    start_time = time.time()

    prompt = making_instructions(config)
    
    # Specify a save folder and save metadata
    model_dir = find_model_dir(config["model_name"])
    folder_path = f'result_txt/{os.path.splitext(config["task_type"])[0]}/{model_dir}_{config["stimuli"]["name"]}'
    if config["overwrite"] and os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok = True)
    
    with open(os.path.join(folder_path, "metadata.json"), 'w') as metadata_file:
        json.dump({
            "task_type" : config["task_type"],
            "model_dir" : model_dir,
            "timestamp" : datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
            "prompt" : prompt.template
        }, metadata_file, indent=4, ensure_ascii=False)

    error_log = {}
    prev_regen_query = set()
    loop_MAX = 5
    
    for attempt in range(loop_MAX):
        failed_query, num_total_query = make_and_save_answers(config, prompt, folder_path)
        removed_query = remove_small_files(folder_path)
        regen_query = set(failed_query) | set(removed_query)

        logger.info(f"Result of {attempt}-th attempt : {num_total_query - len(regen_query)} / {num_total_query}")
        logger.info(f"Queries to be regenrated: {sorted(regen_query)}")

        if not regen_query: break   # If answers are generated for all queries, exit the loop. 

        prev_regen_query = regen_query
        config["overwrite"] = False
    else:
        if regen_query:
            logger.error(f"Queries still need regeneration after {loop_MAX} attempts: {sorted(regen_query)}")
            error_log = {"config": {k: v for k, v in config.items() if k != "model_config"}, "error_queries" : list(regen_query)}
        
    exe_time = time.time() - start_time
    logger.info(f"Task Finish!\tExecution time: {int(exe_time // 3600)}h {int((exe_time % 3600) // 60)}m {exe_time % 60:.2f}s\n")
    
    return error_log

def task_execution_manager():
    with open("datas/stimuli.json", "r", encoding="UTF-8") as stimuli_file:
        stimuli_list = json.load(stimuli_file)
    stimuli_list = sorted(stimuli_list, key = lambda x: x["name"])

    model_list = ["qwen2chat"]
    task_list = [
        "tasks/common_problem_task_prompt.json",
        "tasks/consequences_task_prompt.json",
        "tasks/im_task_prompt.json",
        "tasks/is_task_prompt.json",
        "tasks/js_task_prompt.json",
        "tasks/situation_task_prompt.json",
        "tasks/unusual_task_prompt.json"
        ]

    error_log_file = f"{datetime.now().strftime('%y%m%d_%H%M')}.json"
    error_log = []
    for model_name in model_list:
        model_config = load_model(model_name)
        if model_config is None:
            continue
        
        for task_type, stimuli in itertools.product(task_list, stimuli_list):
            config = {
                "model_name": model_name,
                "model_config": model_config,
                "task_type": task_type,
                "stimuli": stimuli,
                "overwrite": False,
                "example_num": 100,
                "generate_answer_num" : 5,
                "batch_size": 25,
            }
            result = process_task(config)
            if result:
                error_log.append(result)
                with open(os.path.join("logs", error_log_file), 'w') as log_file:
                    json.dump(error_log, log_file, indent=4, ensure_ascii=False)

        del model_config
        torch.cuda.empty_cache()

if __name__ == "__main__":
    task_execution_manager()