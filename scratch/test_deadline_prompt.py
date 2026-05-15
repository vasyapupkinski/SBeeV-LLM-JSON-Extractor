import requests
import json
import time

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def test_deadline():
    # sample_04의 텍스트 (혜움 공고)
    text = """
    ## 마감일
    2026-12-31 (상시채용 아님)
    ## 근무지역
    서울특별시 강남구 테헤란로86길 13
    """
    
    # 프롬프트에 deadline을 명시적으로 요구
    prompt = f"""다음 채용 공고에서 정보를 JSON 형식으로 추출하세요.
반드시 deadline, company, position, tech_stack 필드를 포함해야 합니다.

공고:
{text}"""

    payload = {
        "model": "sbv-llm",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0}
    }

    print("sbv-llm 모델에게 'deadline' 추출 테스트 중...")
    start = time.time()
    resp = requests.post(OLLAMA_API_URL, json=payload)
    latency = time.time() - start
    
    if resp.status_code == 200:
        result = resp.json().get("response", "")
        print(f"\n결과 ({latency:.2f}s):")
        print(result)
    else:
        print(f"에러 발생: {resp.status_code}")

if __name__ == "__main__":
    test_deadline()
