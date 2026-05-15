import json

with open("data/inference_results/wanted_100_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if item["file_name"] in ["353819.md", "355001.md"]:
        print(f"--- {item['file_name']} ---")
        try:
            print(json.dumps(item["extracted_data"], ensure_ascii=False, indent=2)[:500])
        except Exception:
            print(str(item["extracted_data"])[:500])
