import requests
import json
from pathlib import Path
import re

def clean_text(content):
    # 현재 적용된 전처리 로직 그대로 재현
    noise_markers = ["## 이 포지션을 찾고 계셨나요?", "본 채용 정보는", "지원하기"]
    cut_index = len(content)
    for marker in noise_markers:
        idx = content.find(marker)
        if idx != -1:
            cut_index = min(cut_index, idx)
    content = content[:cut_index]
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)
    content = re.sub(r'http[s]?://\S+', '', content)
    content = re.sub(r'\S{100,}', '', content)
    return content.strip()[:6000]

def test_single_file(file_id):
    file_path = Path(f"data/raw/wanted/{file_id}.md")
    print(f"\n--- Testing {file_id}.md ---")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()
    
    clean_content = clean_text(raw_content)
    print(f"[PREPROCESSED LENGTH]: {len(clean_content)} chars")
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "sbv-llm",
        "prompt": clean_content,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
            "repeat_penalty": 1.5,
            "num_ctx": 16384
        }
    }
    
    response = requests.post(url, json=payload)
    output = response.json().get("response", "")
    
    print(f"[OUTPUT LENGTH]: {len(output)} chars")
    print(f"[RAW OUTPUT (First 500)]: {output[:500]}")
    print(f"[RAW OUTPUT (Last 500)]: {output[-500:]}")
    
    # 루프 여부 확인
    if len(output) > 2000:
        # 마지막 부분에 같은 단어가 반복되는지 체크
        last_chars = output[-500:]
        print("\n⚠️ LOOP DETECTION:")
        print(f"   Last 500 chars snippet: {last_chars}")

if __name__ == "__main__":
    test_single_file("354023") # 티냅스
    test_single_file("354221") # 지피에이코리아
