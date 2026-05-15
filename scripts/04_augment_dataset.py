"""
04_augment_dataset.py — 데이터 증강

앞서 생성된 층화 분할(Stratified Split)된 Train 데이터(800개)를 기반으로
모델의 지시어 적응력 및 견고성을 높이기 위한 데이터 증강을 수행합니다.

Val/Test 데이터는 평가용이므로 절대 증강하지 않습니다.

Usage:
    python scripts/04_augment_dataset.py
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 앞서 나눈 train.jsonl만 불러옵니다.
INPUT_PATH = PROJECT_ROOT / "data" / "splits" / "train.jsonl"
# 증강된 결과는 별도의 파일명으로 저장합니다.
OUTPUT_PATH = PROJECT_ROOT / "data" / "splits" / "train_augmented.jsonl"

SEED = 42

INSTRUCTIONS = [
    "다음 채용 공고에서 구조화된 정보를 JSON으로 추출하세요.",
    "아래 텍스트를 분석하여 채용 정보를 JSON 형식으로 변환하시오.",
    "다음 공고에서 회사명, 포지션, 기술스택 등을 JSON으로 정리해 주세요.",
    "주어진 채용 공고의 핵심 정보를 구조화된 JSON 데이터로 추출하세요.",
    "아래 채용 공고를 읽고, 정해진 스키마에 맞춰 JSON으로 출력하세요.",
]

PARTIAL_INST = "다음 공고에서 기술 스택(tech_stack)과 경력 요건(experience)만 추출하세요."

def load_data() -> list[dict]:
    data = []
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} 파일이 없습니다. 분할 스크립트를 먼저 실행하세요.")
        return data
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def augment(data: list[dict]) -> list[dict]:
    augmented = []
    for s in data:
        input_text = s.get("input", "")
        output_text = s.get("output", "{}")
        
        if not isinstance(output_text, str):
            output_text = json.dumps(output_text, ensure_ascii=False)

        # 1) 전체 추출 (원본에 다양한 instruction 무작위 적용)
        augmented.append({
            "instruction": random.choice(INSTRUCTIONS), 
            "input": input_text, 
            "output": output_text
        })

        # 2) 부분 추출 (특정 필드만 요구하는 경우에 대비)
        try:
            full = json.loads(output_text)
            partial = {"tech_stack": full.get("tech_stack", []), "experience": full.get("experience")}
            augmented.append({
                "instruction": PARTIAL_INST, 
                "input": input_text, 
                "output": json.dumps(partial, ensure_ascii=False)
            })
        except json.JSONDecodeError:
            pass

        # 3) 지시어 변형 (50% 확률로 다른 지시어를 사용해 추가 증강)
        if random.random() < 0.5:
            augmented.append({
                "instruction": random.choice(INSTRUCTIONS[1:]), 
                "input": input_text, 
                "output": output_text
            })

    # 4) 엣지 케이스 (내용이 거의 없는 경우 대비 노이즈 주입)
    for _ in range(50):
        augmented.append({
            "instruction": random.choice(INSTRUCTIONS),
            "input": "채용 공고: 자세한 내용은 홈페이지를 참고하세요. 문의: 02-1234-5678",
            "output": json.dumps({"company": None, "position": None, "tech_stack": [], "experience": None, "location": None, "salary_range": None, "employment_type": None}, ensure_ascii=False),
        })

    return augmented

def write_jsonl(path: Path, data: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def main():
    random.seed(SEED)
    data = load_data()
    if not data:
        return
    print(f"원본 Train 데이터: {len(data)}건")

    aug = augment(data)
    random.shuffle(aug)
    
    write_jsonl(OUTPUT_PATH, aug)
    print(f"증강된 Train 데이터: {len(aug)}건")
    print(f"저장 위치: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
