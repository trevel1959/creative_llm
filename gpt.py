import os
import openai
import logging
import json
from tqdm import tqdm
import shutil
import time
import re
import glob
import itertools
import concurrent.futures

def set_logger(level):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s\t- %(message)s', datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def text_models_single(config, task_prompt, query, max_length=1024):
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 원하는 GPT-4 모델 지정
            messages=[
                {"role": "system", "content": f"""주어진 질문을 따라, 창의적이고 독창적인 아이디어와 상세한 설명을 {config["generate_answer_num"]+2}개 생성하세요."""},
                # {"role": "system", "content": f"""For the following questions, please generate {config["generate_answer_num"]+2} CREATIVE and ORIGINAL ideas with detailed explanations for each idea"},
                # {"role": "system", "content": f"""YOU MUST Write the numbers before each answer as follows: "1.", "2."."""},
                {"role": "system", "content": config["stimuli"]["text"]},
                {"role": "user", "content": task_prompt + query},
            ],
            max_tokens=max_length,
            temperature=0.9,
            top_p=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in GPT-4 API call: {e}")
        return ""

def make_and_save_answers_single_parallel(config, folder_path):
    with open(config["task_type"], "r", encoding="UTF-8") as task_file:
        task_all_data = json.load(task_file)
    
    task_prompt = task_all_data["task_prompt"]
    task_data = task_all_data["task_list"]

    example_num = min(len(task_data), config["example_num"])
    task_data = task_data[:example_num]

    processed_queries = []
    failed_queries = []

    def process_query(index, query):
        file_path = os.path.join(folder_path, f"Q{index:02d}.json")

        # 파일이 이미 존재하는 경우, config에서 overwrite가 False이면 스킵
        if not config["overwrite"] and os.path.exists(file_path):
            return False, index

        # GPT-4 API 호출
        inf_result = text_models_single(config, task_prompt, query)
        # print(inf_result)
        parsed_str = parsing_text(inf_result)
        # print(parsed_str)
                    
        if parsed_str and len(parsed_str) >= config["generate_answer_num"] + 2:
            with open(file_path, "w", encoding="UTF-8") as output_file:
                json.dump(parsed_str[1:config["generate_answer_num"] + 1], output_file, indent=4, ensure_ascii=False)
            return True, None  # 성공한 경우
        else:
            return True, index  # 실패한 경우

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_query, index, query) 
            for index, query in enumerate(task_data)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), desc="Generating answers...", total=example_num):
            result, failed_index = future.result()
            if result: processed_queries.append(result)
            if result and (failed_index is not None):
                failed_queries.append(failed_index)

    return failed_queries, processed_queries

def parsing_text_legacy(text): 
    start_idx = text.find("1.")
    if start_idx == -1:
        start_idx = 0
    text = re.sub(r':\s*\n+', ': ', text[start_idx:])   # remove blank and newline after ':' in text
    text = re.sub(r'(?<!\d\.)\n+', ' ', text)
    
    lines = list(filter(lambda x: x.strip(), text.split("\n"))) # split text into lines and remove empty line
    result = []
    for line in lines:
        formatted_line = re.sub(r"^\d+\.\s*", "", line) # remove list numbering for each line
        if line != formatted_line and formatted_line:   # append it to result when there is numbering and not empty
            result.append(formatted_line)

    return result

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

def process_task(logger, config):
    logger.info("Task Start.")
    logger.info(config)
    start_time = time.time()
    
    folder_path = f'result_txt/{os.path.splitext(config["task_type"])[0]}/openai/{config["model_name"]}_{config["stimuli"]["name"]}'
    if config["overwrite"] and os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok = True)
    
    failed_query, num_total_query = make_and_save_answers_single_parallel(config, folder_path)
    num_fail = len(failed_query)
    num_total = len(num_total_query)

    logger.info(f"Result : {num_total - num_fail} / {num_total}")
    logger.info(f"Queries to be regenrated: {sorted(failed_query)}")
        
    exe_time = time.time() - start_time
    logger.info(f"Task Finish!\tExecution time: {int(exe_time // 3600)}h {int((exe_time % 3600) // 60)}m {exe_time % 60:.2f}s\n")
    
    return num_fail, num_total
    
def task_execution_manager(lang):
    logger = set_logger(logging.INFO)
    
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
    model_list = ["gpt-4o-mini"]
    
    while True:
        invalid_result = 0
        
        for model_name in model_list:
            for task_type, stimuli in itertools.product(task_list, stimuli_list):
                config = {
                    "model_name": model_name,
                    "task_type": task_type,
                    "stimuli": stimuli,
                    "overwrite": False,
                    "example_num": 100,
                    "generate_answer_num" : 5,
                    "lang": lang,
                    "batch_size": 1
                }
                
                num_fail, num_total = process_task(logger, config)
                
        # if invalid_result == 0: break
        break
    return
    
def delete_invalid_json_files(folder_path):
    deleted_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        print(root, len(files))
        # for file in files:
        #     if file == 'metadata.json':
        #         file_path = os.path.join(root, file)
        #         print(file_path)
    
        #         os.remove(file_path)

if(__name__ == "__main__"):
    with open("apikey.txt", "r") as apikey_file:
        os.environ["OPENAI_API_KEY"] = apikey_file.read()
    
    task_execution_manager(lang = "ko")
    # delete_invalid_json_files("result_txt\\tasks\\is_task_prompt\\openai")