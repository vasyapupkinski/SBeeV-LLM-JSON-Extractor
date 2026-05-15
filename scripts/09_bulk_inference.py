import os
import json
import requests
import time
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wanted"
OUTPUT_FILE = PROJECT_ROOT / "data" / "inference_results" / "wanted_100_results.json"

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt):
    payload = {
        "model": "sbv-llm",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
            "repeat_penalty": 1.5,
            "repeat_last_n": 1024,
            "num_ctx": 16384
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

    target_files = all_files[:100]
    print(f"[API 모드] {len(target_files)}개의 공고 분석 시작...\n")

    results = []
    start_time = time.time()

    for i, file_path in enumerate(target_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 전처리 1: 데이터 기반 노이즈 절단 (더 정교하게)
            # 1,000개 분석 결과, 실제 공고 본문은 아래 문구들이 나오기 직전에 끝남
            noise_markers = [
                "## 이 포지션을 찾고 계셨나요?",
                "본 채용 정보는",
                "지원하기"
            ]
            
            # 가장 먼저 나타나는 마커를 찾아서 그 지점부터 잘라버림
            cut_index = len(content)
            for marker in noise_markers:
                idx = content.find(marker)
                if idx != -1:
                    cut_index = min(cut_index, idx)
            
            content = content[:cut_index]

            # 전처리 2: 이미지 태그, URL 및 비정상적으로 긴 문자열 제거
            content = re.sub(r'!\[.*?\]\(.*?\)', '', content) # 이미지 태그 제거
            content = re.sub(r'\[.*?\]\(.*?\)', '', content)  # 일반 마크다운 링크도 제거
            content = re.sub(r'http[s]?://\S+', '', content) # 일반 URL 제거
            content = re.sub(r'\S{100,}', '', content)        # 공백 없이 100자 이상 이어지는 기괴한 문자열(트래킹 픽셀 등) 삭제
            
            # 최종 클리닝 및 길이 제한
            clean_content = content.strip()[:6000]
            
            print(f"[{i}/{len(target_files)}] {file_path.name} 처리 중...", end=" ", flush=True)

            output = query_ollama(clean_content)
            
            if output.startswith("ERROR:"):
                print(f"❌ {output}")
                continue

            # JSON 파싱 시도
            try:
                # 모델이 마크다운 블록 안에 넣었을 경우 대비
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
                "extracted_data": parsed_json,
                "error": error_msg if not is_valid else None
            })
            
            if is_valid:
                print("✅")
            else:
                print(f"⚠️ 포맷 에러: {error_msg}")
                # 디버깅을 위해 RAW OUTPUT 전체를 파일로 저장
                debug_path = Path("scratch/debug_failed_output.txt")
                with open(debug_path, "w", encoding="utf-8") as df:
                    df.write(output)
                print(f"   [!] 상세 로그 저장됨: {debug_path}")
                print(f"   [RAW SNIPPET]: {output[:100]}...")
                
        except Exception as e:
            print(f"❌ 중대 에러: {e}")

    # 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "="*50)
    print(f" 100개 테스트 완료! (소요 시간: {elapsed:.1f}초)")
    print(f"건당 평균 속도: {elapsed/len(target_files):.2f}초")
    print(f"저장 위치: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()
