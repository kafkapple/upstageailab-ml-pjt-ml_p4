from dotenv import load_dotenv; import os, json, sys, datetime

envFolder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(f'{envFolder}/.env')
client_id, client_secret = os.getenv("naverclient"), os.getenv("naversecret")

# 콘솔 출력 인코딩을 UTF-8로 설정
sys.stdout.reconfigure(encoding='utf-8')

url = "https://openapi.naver.com/v1/search/news.json"
headers = {"X-Naver-Client-Id":client_id, "X-Naver-Client-Secret":client_secret}
params = { "query" : "대한민국", "display" : 1}

import requests as req; res = req.get(url, headers=headers, params=params)

# 저장할 디렉토리와 파일 경로 설정
save_dir = "data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, "naver_news_result.json")

if(res.status_code==200):
    new_items = res.json()['items']  # 새로운 뉴스 항목들
    
    # 기존 뉴스 읽기 (없으면 빈 리스트)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_items = json.load(f)
            # saved_items가 리스트가 아닌 경우, 빈 리스트로 초기화
            if not isinstance(saved_items, list):
                saved_items = []
    except (json.JSONDecodeError, FileNotFoundError):
        saved_items = []
    
    existing_links = {item['link'] for item in saved_items}
    new_unique_items = [item for item in new_items if item['link'] not in existing_links]
    saved_items.extend(new_unique_items)
    
    # 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(saved_items, f, ensure_ascii=False, indent=2)
    
    # 새로 받은 뉴스만 출력
    print(json.dumps(new_items, ensure_ascii=False, indent=2))
else:
    print(f"Error Code: {res.status_code}")
# %%
