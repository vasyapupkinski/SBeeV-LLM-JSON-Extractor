import os
import json
import requests
import time
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wanted"
OUTPUT_FILE = PROJECT_ROOT / "data" / "inference_results" / "wanted_v2_results.json"

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def clean_text(text):
    # 1. Markdown 이미지 링크 제거 (![alt](url))
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 2. 일반 링크에서 URL 제거 ([text](url) -> text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # 3. 노이즈 마커 이후 제거
    noise_patterns = [r"## 이 포지션을 찾고 계셨나요\?", r"본\s*채용\s*정보는", r"지원하기", r"서류 합격 확률이 아주 높아요"]
    cut_index = len(text)
    for pattern in noise_patterns:
        match = re.search(pattern, text)
        if match:
            cut_index = min(cut_index, match.start())
    text = text[:cut_index].strip()
    # 4. 불필요한 공백 및 특수문자 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text[:6000] # 모델 입력 길이 제한

def query_ollama(prompt):
    payload = {
        "model": "sbv-llm-v2", # V2 모델 사용
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 1024,
            "repeat_penalty": 2.0
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def main():
    if not RAW_DIR.exists():
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {RAW_DIR}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(list(RAW_DIR.glob("*.md")))
    if not all_files:
        print("❌ 마크다운 파일이 없습니다.")
        return

    # 사용자 요청에 따라 100개 테스트
    target_files = all_files[:100]
    print(f"[V2 테스트 모드] {len(target_files)}개의 공고 분석 시작 (Model: sbv-llm-v2)...\n")

    results = []
    start_time = time.time()

    for i, file_path in enumerate(target_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 강력한 정제 함수 적용
            cleaned_content = clean_text(content)
            
            print(f"[{i}/{len(target_files)}] {file_path.name} 처리 중...", end=" ", flush=True)

            # 명시적 지시어 추가 (모델이 SYSTEM 프롬프트를 더 잘 따르도록 유도)
            full_prompt = f"다음 채용 공고에서 정보를 추출하여 JSON으로 출력하세요:\n\n{cleaned_content}"
            output = query_ollama(full_prompt)
            
            if output.startswith("ERROR:"):
                print(f"❌ {output}")
                continue

            try:
                json_str = output
                if "```json" in output:
                    json_str = output.split("```json")[1].split("```")[0].strip()
                elif "```" in output:
                    json_str = output.split("```")[1].split("```")[0].strip()
                
                parsed_json = json.loads(json_str)
                is_valid = True
            except Exception as e:
                parsed_json = output
                is_valid = False
                error_msg = str(e)

            results.append({
                "file_name": file_path.name,
                "is_valid_json": is_valid,
                "extracted_data": parsed_json if is_valid else None,
                "raw_output": output, # 모델이 내뱉은 원문 그대로 저장
                "error": error_msg if not is_valid else None
            })
            
            if is_valid:
                print("✅")
            else:
                print(f"⚠️ 에러: {error_msg}")
                # 에러 난 경우 raw output도 로그용으로 남겨두면 좋음
                
            # 실시간 저장 (하나 끝날 때마다 업데이트)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ 중대 에러 ({file_path.name}): {e}")

    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print(f" V2 테스트 완료! (총 {len(results)}개 처리, 소요 시간: {elapsed:.1f}초)")
    print(f"최종 저장 위치: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()
