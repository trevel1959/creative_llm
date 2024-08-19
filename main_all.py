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

def text_models_batch(model_config, input_strs, max_length=1024):
    tokenizer, model, device = model_config
    model.eval()
    
    # 입력 문자열을 배치 단위로 토크나이즈
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

def making_instructions(generate_answer_num, model_name, stimuli, user_query, lang):
    prompt_format = {
        "llama2chat": Template("""<s>[INST] <<SYS>>$system_prompt$system_stimuli\n<</SYS>>\n\n$user_query [/INST]"""),  # https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/
        "qwenchat": Template("""<s>$system_prompt$system_stimuli\n$user_query"""),   # temp
        "mistralinst": Template("""<s>[INST] $system_prompt$system_stimuli\n# question:\n$user_query [/INST]"""),   # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/discussions/49
        "default": Template("""<s>[INST] $system_prompt$system_stimuli\n$user_query [/INST]""") # temp
    }
    system_prompt = {
        "en": f"""\nFor the following questions, generate {generate_answer_num} CREATIVE and ORIGINAL ideas with detailed explanations.""",
        "ko": f"""\n주어진 질문을 따라, {generate_answer_num}개의 창의적이고 독창적인 아이디어를 상세한 설명과 함께 생성하세요.""",
    }
    system_stimuli = stimuli
    
    prompt_temp = prompt_format.get(model_name, prompt_format["default"]).safe_substitute(
        system_prompt = system_prompt[lang],
        system_stimuli = system_stimuli,
        user_query = user_query
    )

    return prompt_temp

def process_task(config, model_config):
    logger = set_logger(logging.INFO)
    
    logger.info("Task Start.")
    start_time = time.time()
    
    #####################################################################################################################
    # collect batch datas
    model_dir = find_model_dir(config["model_name"])
    
    batch_data = []
    batch_size = config["batch_size"]
    for task_type, stimuli in itertools.product(config["task_list"], config["stimuli_list"]):
        folder_path = f'result_txt/{os.path.splitext(task_type)[0]}/{model_dir}_{stimuli["name"]}'
        
        with open(task_type, "r", encoding="UTF-8") as task_file:
            task_all_data = json.load(task_file)
    
        task_prompt = task_all_data["task_prompt"]
        task_data = task_all_data["task_list"]
        
        example_num = min(len(task_data), config["example_num"])
        
        for i, task in enumerate(task_data[:example_num]):
            file_path = os.path.join(folder_path, f"Q{i:02d}.json")
            batch_input = making_instructions(config["generate_answer_num"], config["model_name"], stimuli["text"], task_prompt + task, config["lang"])
            batch_data.append((batch_input, file_path))
    
    if not config["overwrite"]:
        batch_data = [data for data in batch_data if not os.path.exists(data[1])]
    
    # inference queries
    failed_queries = []
    if batch_data:
        for i in tqdm(range(0, len(batch_data), batch_size), desc="Generating answers...", total=math.ceil(len(batch_data) / batch_size)):
            batch = batch_data[i:i + batch_size]
            batch_queries, batch_file_paths = zip(*batch)

            inf_results = text_models_batch(model_config, batch_queries)
            parsed_strs = [parsing_text(result) for result in inf_results]
                    
            for parsed_str, file_path, index in zip(parsed_strs, batch_file_paths, batch_indices):
                if len(parsed_str) >= config["generate_answer_num"]:
                    with open(file_path, "w", encoding="UTF-8") as output_file:
                        json.dump(parsed_str[:config["generate_answer_num"]], output_file, indent=4, ensure_ascii=False)
                else:
                    failed_queries.append(index)

    logger.info(failed_queries)

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
    
    with open(stimuli_file_path, "r", encoding="UTF-8") as stimuli_file:
        stimuli_list = json.load(stimuli_file)
    stimuli_list = sorted(stimuli_list, key = lambda x: x["name"])
    task_list = sorted(glob.glob(f'{task_folder_path}/*'))
    model_list = ["llama3.1inst"]

    error_log_file = f"{datetime.now().strftime('%y%m%d_%H%M')}.json"
    error_log = []
    for model_name in model_list:
        model_config = load_model(model_name)
        if model_config is None:
            continue
        
        config = {
            "model_name": model_name,
            "task_list": task_list,
            "stimuli_list": stimuli_list,
            "overwrite": False,
            "example_num": 100,
            "generate_answer_num" : 5,
            "batch_size": 10,
            "lang": "ko",
        }
        result = process_task(config, model_config)

        del model_config
        torch.cuda.empty_cache()

if __name__ == "__main__":
    task_execution_manager(lang = "ko")