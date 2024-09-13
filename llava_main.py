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

from tqdm import tqdm
from datetime import timedelta, datetime
from string import Template
from transformers import pipeline

model_ids = {
    "llava-mistral": "llava-hf/llava-v1.6-mistral-7b-hf", # base - mistral inst 0.2
    "llava-vicuna": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llava-llama3": "llava-hf/llama3-llava-next-8b-hf" # base- llama3 inst
}

def load_pipe(model_name):
    model_id = model_ids[model_name]

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-to-text", model=model_id, device = device)

def llava_inference_simple(pipe, img, prompt = "User: hello! \nAssistant:", max_new_tokens = 1024):
    outputs = pipe(
        images = img,
        prompt = "<image>" + prompt,
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.7
        }
    )
    return outputs[0]["generated_text"]

def llava_inference_batch(pipe, img, prompts, max_new_tokens = 1024):
    imgs = [img] * len(prompts)
    prompts = ["<image>"+prompt for prompt in prompts]
    
    outputs = pipe(
        images = imgs,
        prompt = prompts,
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.9,
            "top_p": 0.7
        }
    )
    return [output["generated_text"] for output in outputs]

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

def text_models_batch(model_config, img, prompt, task_prompt, queries, max_length=512):
    pipe = model_config
    
    # 입력 문자열을 배치 단위로 토크나이즈
    input_strs = [prompt.safe_substitute(user_query= task_prompt + query) + "\n1." for query in queries]
    output_texts = []
    for input in input_strs:
        output_texts.append(llava_inference_simple(pipe, img, input))

    return output_texts
    # 출력 토큰을 텍스트로 변환
    # return llava_inference_batch(pipe, img, input_strs)

def make_and_save_answers(config, model_config, prompt, folder_path):
    with open(config["task_type"], "r", encoding="UTF-8") as task_file:
        task_all_data = json.load(task_file)
    
    task_prompt = task_all_data["task_prompt"]
    task_data = task_all_data["task_list"]
    
    img = config["image_dir"]

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

            inf_results = text_models_batch(model_config, img, prompt, task_prompt, batch_queries)
            parsed_strs = [parsing_text(result) for result in inf_results]
            # for str in parsed_strs:
            #     print(str)
            # print("--------------------------------------------------------------------------------")
                    
            for parsed_str, file_path, index in zip(parsed_strs, batch_file_paths, batch_indices):
                if len(parsed_str) >= config["generate_answer_num"]+2:
                    with open(file_path, "w", encoding="UTF-8") as output_file:
                        json.dump(parsed_str[1:config["generate_answer_num"]+1], output_file, indent=4, ensure_ascii=False)
                else:
                    failed_queries.append(index)

    return failed_queries, len(batch_data)

def parsing_text(text):
    # (1) 줄바꿈을 제외한 모든 줄바꿈을 공백으로 치환
    text = re.sub(r'(?<!\d\.)\n+', ' ', text)
    
    # (2) 넘버링을 기준으로 문자열 분리
    # "\n"을 넘버링 뒤에 붙여서 다음 넘버링 시작 전에 줄바꿈을 넣어줌으로써 구분하기 쉽게 함
    text = re.sub(r'(\d+)\.\s', r'\n\1. ', text)
    
    # (3) 넘버링으로 시작하는 문구만 추출 (리스트로 변환)
    sections = re.findall(r'\d+\.\s[^\n]+', text)

    # (4) 넘버링 제거
    result = [re.sub(r'\s+', ' ', re.sub(r'^\d+\.\s*', '', section)).strip() for section in sections]

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
        "llava-llama3": Template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$system_prompt<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n$user_query<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""),
        "llava-mistral": Template("""<s>[INST] $system_prompt$system_stimuli\n# question:\n$user_query [/INST]"""),   # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/discussions/49
        "default": Template("""$system_prompt$system_stimuli\nUSER:\n$user_query""") # temp
    }
    system_prompt = {
        "en": f"""\nFor the following questions, generate {config["generate_answer_num"]+2} CREATIVE and ORIGINAL ideas with detailed explanations.""",
        "ko": f"""\n주어진 질문을 따라, {config["generate_answer_num"]+2}개의 창의적이고 독창적인 아이디어를 상세한 설명과 함께 생성하세요. 답변은 반드시 한국어로 하세요.""",
        "cn": f"""\n根据所给的问题，生成{config["generate_answer_num"]+2}个具有创造性和独特性的想法，并附上详细说明。答案必须用中文书写。"""
    }
    system_stimuli = config["stimuli"]["text"]
    
    prompt_temp = prompt_format.get(config["model_name"], prompt_format["default"]).safe_substitute(
        system_prompt = system_prompt[config["lang"]],
        system_stimuli = system_stimuli
    )

    return Template(prompt_temp)

def process_task(config, model_config):
    logger = set_logger(logging.INFO) #######################################################################################################################################################################
    
    logger.info("Task Start.")
    logger.info(config)
    start_time = time.time()

    prompt = making_instructions(config)
    
    # Specify a save folder
    model_dir = find_model_dir(config["model_name"])
    if config["image_dir"] is not None:
        folder_path = f'result_txt/{os.path.splitext(config["task_type"])[0]}/{model_dir}_{config["image_dir"][7:13]}'
    else:
        folder_path = f'result_txt/{os.path.splitext(config["task_type"])[0]}/{model_dir}_{config["stimuli"]["name"]}'
    if config["overwrite"] and os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok = True)

    error_log = {}
    prev_regen_query = set()
    loop_MAX = 1
    
    for attempt in range(loop_MAX):
        failed_query, num_total_query = make_and_save_answers(config, model_config, prompt, folder_path)
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
            error_log = {"config": config, "error_queries" : list(regen_query)}
        
    exe_time = time.time() - start_time
    logger.info(f"Task Finish!\tExecution time: {int(exe_time // 3600)}h {int((exe_time % 3600) // 60)}m {exe_time % 60:.2f}s\n")
    
    return error_log

# def contains_korean(text):
#     # 한글 유니코드 범위: 가-힣
#     korean_pattern = re.compile(r'[가-힣]')
#     return bool(korean_pattern.search(text))

# def delete_json_if_contains_korean(file_path):
#     try:
#         # JSON 파일 열기
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data = json.load(file)

#         # JSON 데이터가 리스트인 경우에만 처리
#         if isinstance(data, list):
#             # 문자열 리스트에 한글이 포함된 항목이 있는지 확인
#             for s in data:
#                 if isinstance(s, str) and contains_korean(s):
#                     print(f"한글이 포함된 문자열이 발견되어 파일을 삭제합니다: {file_path}")
#                     os.remove(file_path)  # 파일 삭제
#                     return
#             # print(f"한글이 포함된 문자열이 없으므로 파일을 유지합니다: {file_path}")
#         else:
#             print(f"JSON 데이터가 리스트가 아닙니다. 스킵합니다: {file_path}")
#     except Exception as e:
#         print(f"오류가 발생했습니다: {e}")

# def process_all_json_files_in_folder(root_folder):
#     # 폴더 내 모든 파일과 하위 폴더를 재귀적으로 순회
#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.endswith('.json'):
#                 file_path = os.path.join(dirpath, filename)
#                 delete_json_if_contains_korean(file_path)

def task_execution_manager(lang):
    if lang == "ko":
        stimuli_file_path = "datas/stimuli_ko.json"
        task_folder_path = "tasks_ko"
    elif lang == "cn":
        stimuli_file_path = "datas/stimuli_cn.json"
        task_folder_path = "tasks_cn"
    else:
        stimuli_file_path = "datas/stimuli.json"
        task_folder_path = "tasks"
    
    with open(stimuli_file_path, "r", encoding="UTF-8") as stimuli_file:
        stimuli_list = json.load(stimuli_file)
    stimuli_list = sorted(stimuli_list, key = lambda x: x["name"])
    stimuli_list = [stimuli_list[0]]    ######################################################################################
    
    image_list = sorted(glob.glob('images/*')) + [None]
    print(image_list)
    task_list = sorted(glob.glob(f'{task_folder_path}/*'))
    model_list = ["llava-mistral"]

    error_log_file = f"{datetime.now().strftime('%y%m%d_%H%M')}.json"
    error_log = []
    
    for model_name in model_list:
        model_config = load_pipe(model_name)
        if model_config is None:
            continue
        
        for task_type, stimuli, image_dir in itertools.product(task_list, stimuli_list, image_list):
            config = {
                "model_name": model_name,
                "task_type": task_type,
                "stimuli": stimuli,
                "image_dir": image_dir,
                "overwrite": False,
                "example_num": 100,
                "generate_answer_num" : 5,
                "batch_size": 1,
                "lang": lang,
            }
            result = process_task(config, model_config)

        del model_config
        torch.cuda.empty_cache()

if __name__ == "__main__":
    task_execution_manager(lang = "en")