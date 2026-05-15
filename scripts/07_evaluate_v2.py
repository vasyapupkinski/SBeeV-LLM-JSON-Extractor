import json
import os
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "splits_v2" / "test.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results_v2"
LOG_DIR = RESULTS_DIR / "eval_logs"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

FIELDS = ["company", "position", "tech_stack", "experience", "location", "deadline", "salary_range", "employment_type"]

def calculate_f1(gold_list, pred_list):
    if not gold_list and not pred_list: return 1.0, 1.0, 1.0
    if not pred_list: return 0.0, 0.0, 0.0
    gold_set = set([s.lower().strip() for s in (gold_list or [])])
    pred_set = set([s.lower().strip() for s in (pred_list or [])])
    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def unload_model(model_name):
    try:
        requests.post(OLLAMA_API_URL, json={"model": model_name, "keep_alive": 0})
    except:
        pass

import re

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
    return text[:6000]

def infer_ollama(text: str, model_name: str) -> tuple[dict | None, float]:
    # 강력한 정제 함수 적용
    cleaned_text = clean_text(text)

    prompt = f"당신은 정보 추출기입니다. 절대 같은 단어를 반복하지 마세요.\n다음 채용 공고에서 8개 필드(company, position, tech_stack, experience, location, deadline, salary_range, employment_type)를 JSON으로 추출하세요.\n\n공고:\n{cleaned_text}"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0, 
            "num_predict": 1024, 
            "repeat_penalty": 2.0
        }
    }
    start = time.time()
    try:
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        if resp.status_code != 200:
            return None, time.time() - start
        output = resp.json().get("response", "").strip()
        try:
            return json.loads(output), time.time() - start
        except:
            return None, time.time() - start
    except Exception as e:
        return None, time.time() - start

def infer_openai(text: str) -> tuple[dict | None, float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return None, 0
    client = OpenAI(api_key=api_key)
    prompt = f"다음 채용 공고에서 8개 필드를 JSON으로 추출하세요. (company, position, tech_stack, experience, location, deadline, salary_range, employment_type)\n\n공고:\n{text[:8000]}"
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(resp.choices[0].message.content), time.time() - start
    except Exception as e:
        return None, time.time() - start

def run_eval(samples: list[dict], infer_fn, name: str, model_to_unload=None) -> dict:
    print(f"\nEvaluating: {name}...")
    f1_scores = []
    latencies = []
    valid_count = 0
    target_count = min(100, len(samples)) # 100개로 상향
    
    model_log_dir = LOG_DIR / name.replace(" ", "_").replace("(", "").replace(")", "")
    model_log_dir.mkdir(parents=True, exist_ok=True)
    
    for i, sample in enumerate(samples[:target_count]):
        gold = json.loads(sample["output"])
        pred, latency = infer_fn(sample["input"])
        latencies.append(latency)
        print(f"  [{i+1}/{target_count}] Processing... ({latency:.2f}s)", end="\r")
        
        log_file = model_log_dir / f"sample_{i+1:02d}.json"
        log_data = {"model": name, "gold": gold, "pred": pred, "latency": latency}
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        if pred:
            valid_count += 1
            _, _, f1 = calculate_f1(gold.get("tech_stack", []), pred.get("tech_stack", []))
            f1_scores.append(f1)
        
    if model_to_unload: unload_model(model_to_unload)
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print(f" -> F1: {avg_f1:.4f}, Latency: {avg_lat:.2f}s, Valid: {valid_count}/{target_count}")
    return {"f1": avg_f1, "latency": avg_lat, "valid_rate": valid_count / target_count}

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    all_results = {}
    gpt_res = run_eval(samples, infer_openai, "GPT-4o-mini")
    if gpt_res["valid_rate"] > 0: all_results["GPT-4o-mini"] = gpt_res
    
    qwen_res = run_eval(samples, lambda t: infer_ollama(t, "qwen3.5:9b"), "Qwen 3.5 (Base)", model_to_unload="qwen3.5:9b")
    all_results["Qwen3.5-Base"] = qwen_res
    
    sbv_res = run_eval(samples, lambda t: infer_ollama(t, "sbv-llm-v2"), "SBV-LLM-V2 (DoRA)", model_to_unload="sbv-llm-v2")
    all_results["SBV-LLM-V2"] = sbv_res
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "final_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
