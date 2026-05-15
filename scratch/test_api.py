import requests
res = requests.post("http://localhost:11434/api/generate", json={
    "model": "sbv-llm", 
    "prompt": "테스트: 슈퍼센트에서 데이터 분석가를 모집합니다.", 
    "stream": False, 
    "format": "json", 
    "options": {"temperature": 0.1}
})
print(res.json().get("response"))
