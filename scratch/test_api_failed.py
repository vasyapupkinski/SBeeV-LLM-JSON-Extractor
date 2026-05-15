import json, requests
with open("data/raw/wanted/354671.md", "r", encoding="utf-8") as f: content = f.read().strip()[:2000]
res = requests.post("http://localhost:11434/api/generate", json={"model": "sbv-llm", "prompt": content, "stream": False, "format": "json", "options": {"temperature": 0.3, "repeat_penalty": 1.5, "num_predict": 1024}})
output = res.json().get("response", "")
print("RAW OUTPUT END:")
print(output[-500:])
try:
    json.loads(output)
    print("✅ Valid JSON")
except Exception as e:
    print(f"❌ JSON Error: {e}")
