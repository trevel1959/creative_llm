import torch
import json
import os
import logging
import time
import re
import shutil
import itertools

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

def load_model(model_dir):
    accelerator = Accelerator()
    device = accelerator.device
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True    # for qwen, 240731 1805KZ
        )
    # 패딩 토큰이 없으면 추가
    if tokenizer.pad_token is None:
        # 패딩 토큰 추가
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        tokenizer.pad_token_id = tokenizer.get_vocab()['[PAD]']
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code = True    # for qwen, 240731 1723KZ
    )
    model.resize_token_embeddings(len(tokenizer))
    model = accelerator.prepare(model)
    
    return tokenizer, model, device

def text_models(model_config, prompt, query):
    tokenizer, model, device = model_config
    model.eval()
    

    
    input_str = prompt.safe_substitute(user_query = query) + " Assistant: \n1."
    input_to_model = tokenizer(input_str, return_tensors="pt").to(device)

    # 모델에 입력을 주고 추론 수행
    with torch.no_grad():
        outputs = model.generate(
            **input_to_model,
            pad_token_id = tokenizer.eos_token_id,  # for mistral, 240731 1034KZ
            max_new_tokens = 512,  # for qwen, 240731 1803KZ
            do_sample = True,
            temperature = 0.9,
            top_p = 0.7
        )

    # 출력 토큰을 텍스트로 변환
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def generate_answers(model_config, config, prompt, folder_path):
    with open(config["task_type"], "r", encoding="UTF-8") as input_file:
        input_data = json.load(input_file)
    
    # Inference and save texts
    loop_MAX = 3
    example_num = min(len(input_data), config["example_num"])
    failed_query = []
    for i, query in tqdm(enumerate(input_data[:example_num]), desc=f"Generating answers...", total = example_num):
        file_path = os.path.join(folder_path, f"Q{i:02d}.json")
        if not config["overwrite"] and os.path.exists(file_path):
            continue
        
        for _ in range(loop_MAX):
            inf_result = text_models(model_config, prompt, query)
            parsed_str = parsing_text(inf_result)

            if len(parsed_str) >= config["generate_answer_num"]:
                with open(file_path, "w", encoding="UTF-8") as output_file:
                    json.dump(parsed_str[:config["generate_answer_num"]], output_file, indent = 4, ensure_ascii = False)
                break
        else:
            failed_query.append(i)
    
    return failed_query

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
        "qwenchat": Template("""<s>[INST] $system_prompt$system_stimuli\n$user_query [/INST]"""),   # temp
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

def main(config):
    logger = set_logger(logging.INFO)
    
    with open("list_model.json", "r", encoding="UTF-8") as model_file:
        models = json.load(model_file)
    
    model_dir = models.get(config["model_name"])
    if model_dir is None:
        logger.error(f'[ERROR] No model corresponds to {config["model_name"]}')
        return
    
    logger.info("Task start.")
    logger.info(config)
    start_time = time.time()
    
    # Load model
    model_config = load_model(model_dir)
    prompt = making_instructions(config)
    
    # Specify a save folder and save metadata
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

    prev_total_query = set()
    loop_MAX = 3
    loop_curr = 0
    
    while loop_curr < loop_MAX:
        failed_query = generate_answers(model_config, config, prompt, folder_path)
        removed_query = remove_small_files(folder_path)
        total_query = set(failed_query) | set(removed_query)

        logger.info(f"Failed queries: {failed_query}")
        logger.info(f"Removed queries: {removed_query}")
        logger.info(f"Num of failed queries: {len(failed_query)}")
        logger.info(f"Num of removed queries: {len(removed_query)}")
        logger.info(f"Num of queries to be recreated: {len(total_query)}")

        if not total_query: break   # If answers are generated for all queries, exit the loop. 

        loop_curr = int(total_query == prev_total_query) * (loop_curr + 1)   # True -> loop_curr += 1, False -> loop_curr = 0
        prev_total_query = total_query
        config["overwrite"] = False
    else:
        logger.error(f"[ERROR] Queries don't change : {total_query}")
        
    logger.info("Task finish!")
    exe_time = time.time() - start_time
    logger.info(f"execution time: {int(exe_time // 3600)}h {int((exe_time % 3600) // 60)}m {exe_time % 60:.2f}s")
    
    del model_config
    torch.cuda.empty_cache()

if __name__ == "__main__":
    with open("list_stimuli.json", "r", encoding="UTF-8") as stimuli_file:
        stimuli_list = json.load(stimuli_file)
    stimuli_list = sorted(stimuli_list, key = lambda x: x["name"])

    model_list = ["mistralinst"]
    
    for model_name, stimuli in itertools.product(model_list, stimuli_list):
        config = {
            "model_name" : model_name,
            "stimuli": stimuli,
            "example_num": 100,
            "generate_answer_num" : 5,
            "overwrite": True,
            "task_type": "tasks/situation_task_prompt.json"
        }
        
        main(config)