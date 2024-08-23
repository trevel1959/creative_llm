import torch
import json
import os
import logging
import time
import re
import shutil
import itertools
import math
import glob

from transformers import LlavaNextForConditionalGeneration, AutoTokenizer
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
        padding_side='left'
        )
        
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_dir,
        token = hf_token,
        device_map="auto",
    )
    
    model = accelerator.prepare(model)
    
    return tokenizer, model, device

def text_models_batch(model_config, prompt, task_prompt, queries, max_length=1024):
    tokenizer, model, device = model_config
    model.eval()
    
    # 입력 문자열을 배치 단위로 토크나이즈
    input_strs = [prompt.safe_substitute(user_query= task_prompt + query) + " Assistant: \n1." for query in queries]
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

def make_and_save_answers(config, model_config, prompt, folder_path):
    with open(config["task_type"], "r", encoding="UTF-8") as task_file:
        task_all_data = json.load(task_file)
    
    task_prompt = task_all_data["task_prompt"]
    task_data = task_all_data["task_list"]
    
    # divide task prompt and data here 240818 0134KZ

    example_num = min(len(task_data), config["example_num"])
    batch_data = [(task_data[i], os.path.join(folder_path, f"Q{i:02d}.json"), i) for i in range(example_num)]
    batch_size = config["batch_size"]

    if not config["overwrite"]:
        batch_data = [data for data in batch_data if not os.path.exists(data[1])]
        
    failed_queries = []
    if batch_data:
        for i in tqdm(range(0, len(batch_data), batch_size), desc="Generating answers...", total=math.ceil(len(batch_data) / batch_size)):
            batch = batch_data[i:i + batch_size]
            batch_queries, batch_file_paths, batch_indices = zip(*batch)

            inf_results = text_models_batch(model_config, prompt, task_prompt, batch_queries)
            parsed_strs = [parsing_text(result) for result in inf_results]
            print(parsed_strs)
                    
            for parsed_str, file_path, index in zip(parsed_strs, batch_file_paths, batch_indices):
                if len(parsed_str) >= config["generate_answer_num"]+2:
                    with open(file_path, "w", encoding="UTF-8") as output_file:
                        json.dump(parsed_str[1:config["generate_answer_num"]+1], output_file, indent=4, ensure_ascii=False)
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

def remove_small_files(folder_path, size_limit_bytes = 1024):
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
        "llava-llama3": Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$system_prompt<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n$user_query<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""),
        "llava-mistral": Template("""<s>[INST] $system_prompt$system_stimuli\n# question:\n$user_query [/INST]"""),   # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/discussions/49
        "default": Template("""$system_prompt$system_stimuli\nUSER:\n$user_query""") # temp
    }
    system_prompt = {
        "en": f"""\nFor the following questions, generate {config["generate_answer_num"]+2} CREATIVE and ORIGINAL ideas with detailed explanations.""",
        "ko": f"""\n주어진 질문을 따라, {config["generate_answer_num"]+2}개의 창의적이고 독창적인 아이디어를 상세한 설명과 함께 생성하세요.""",
    }
    system_stimuli = config["stimuli"]["text"]
    
    prompt_temp = prompt_format.get(config["model_name"], prompt_format["default"]).safe_substitute(
        system_prompt = system_prompt[config["lang"]],
        system_stimuli = system_stimuli
    )

    return Template(prompt_temp)

def process_task(config, model_config):
    logger = set_logger(logging.INFO)
    
    logger.info("Task Start.")
    logger.info(config)
    start_time = time.time()

    prompt = making_instructions(config)
    
    # Specify a save folder
    model_dir = find_model_dir(config["model_name"])
    folder_path = f'result_txt/{os.path.splitext(config["task_type"])[0]}/{model_dir}_{config["stimuli"]["name"]}'
    if config["overwrite"] and os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok = True)
    
    ### make metadata
    # with open(os.path.join(folder_path, "metadata.json"), 'w') as metadata_file:
    #     json.dump({
    #         "task_type" : config["task_type"],
    #         "model_dir" : model_dir,
    #         "timestamp" : datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
    #         "prompt" : prompt.template
    #     }, metadata_file, indent=4, ensure_ascii=False)

    error_log = {}
    prev_regen_query = set()
    loop_MAX = 1
    
    for attempt in range(loop_MAX):
        failed_query, num_total_query = make_and_save_answers(config, model_config, prompt, folder_path)
        # removed_query = remove_small_files(folder_path)
        removed_query = []
        regen_query = set(failed_query) | set(removed_query)

        logger.info(f"Result of {attempt}-th attempt : {num_total_query - len(regen_query)} / {num_total_query}")
        logger.info(f"Queries to be regenrated: {sorted(regen_query)}")

        if not regen_query: break   # If answers are generated for all queries, exit the loop. 

        prev_regen_query = regen_query
        config["overwrite"] = False
    else:
        if regen_query:
            logger.error(f"Queries still need regeneration after {loop_MAX} attempts: {sorted(regen_query)}")
            error_log = {"config": config, "error_queries" : list(regen_query)}
        
    exe_time = time.time() - start_time
    logger.info(f"Task Finish!\tExecution time: {int(exe_time // 3600)}h {int((exe_time % 3600) // 60)}m {exe_time % 60:.2f}s\n")
    
    return error_log

def task_execution_manager(lang = "en"):
    if lang == "ko":
        stimuli_file_path = "datas/stimuli_ko.json"
        task_folder_path = "tasks_ko"
    else:
        stimuli_file_path = "datas/stimuli.json"
        task_folder_path = "tasks"
    
    # with open(stimuli_file_path, "r", encoding="UTF-8") as stimuli_file:
    #     stimuli_list = json.load(stimuli_file)
    # stimuli_list = sorted(stimuli_list, key = lambda x: x["name"])
    stimuli_list = [{'name': 'base', 'text': ''}]
    task_list = sorted(glob.glob(f'{task_folder_path}/*'))
    model_list = ["llava-llama3", "llava-mistral", "llava-vicuna"]

    error_log_file = f"{datetime.now().strftime('%y%m%d_%H%M')}.json"
    error_log = []
    for model_name in model_list:
        model_config = load_model(model_name)
        if model_config is None:
            continue
        
        for task_type, stimuli in itertools.product(task_list, stimuli_list):
            config = {
                "model_name": model_name,
                "task_type": task_type,
                "stimuli": stimuli,
                "overwrite": False,
                "example_num": 100,
                "generate_answer_num" : 5,
                "batch_size": 34,
                "lang": lang,
            }
            result = process_task(config, model_config)
            
            if result:
                error_log.append(result)
                with open(os.path.join("logs", error_log_file), 'w') as log_file:
                    json.dump(error_log, log_file, indent=4, ensure_ascii=False)

        del model_config
        torch.cuda.empty_cache()

if __name__ == "__main__":
    task_execution_manager(lang = "en")