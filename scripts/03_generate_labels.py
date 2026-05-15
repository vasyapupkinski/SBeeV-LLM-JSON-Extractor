"""
03_generate_labels.py — OpenAI GPT-4o-mini 기반 JSON 정답 생성

수집된 원문(.md)에 대해 GPT-4o-mini로 고품질 JSON 정답을 생성한다.
생성된 데이터는 [instruction, input, output] 형식의 JSONL로 저장되어 학습에 사용된다.

Usage:
    python scripts/03_generate_labels.py
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── 설정 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PROJECT_ROOT / "configs" / "schema.json"
RAW_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "wanted",
]
OUTPUT_PATH = PROJECT_ROOT / "data" / "labeled" / "labeled_dataset.jsonl"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "labeled" / "label_checkpoint.json"

# OpenAI 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    JOB_SCHEMA = json.load(f)

SYSTEM_PROMPT = f"""당신은 채용 공고에서 구조화된 정보를 추출하는 전문가입니다.
주어진 채용 공고 텍스트에서 다음 필드를 추출하여 JSON으로 출력하세요.
반드시 아래의 JSON 스키마 형식을 엄격히 준수해야 합니다.

추출 필드:
- company: 회사명 (필수)
- position: 채용 포지션 (필수)
- tech_stack: 기술 스택 리스트 (없으면 빈 배열 [])
- experience: 경력 요건 (예: "3년 이상", "신입", "5-10년")
- location: 근무 지역 (없으면 null)
- salary_range: 연봉 범위 (없으면 null)
- employment_type: 고용 형태 (예: "정규직", "계약직", "인턴")

규칙:
- 반드시 JSON 데이터만 반환하세요.
- 데이터가 없는 필드는 null을 반환하세요 (tech_stack은 []).
- 스키마 구조: {json.dumps(JOB_SCHEMA, ensure_ascii=False)}"""

INSTRUCTION_TEMPLATE = "다음 채용 공고에서 구조화된 정보를 JSON으로 추출하세요."

# ── 체크포인트 ────────────────────────────────────────
def load_checkpoint() -> set:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f).get("processed_files", []))
    return set()

def save_checkpoint(processed: set):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"processed_files": list(processed)}, f, ensure_ascii=False)

# ── 파일 수집 ─────────────────────────────────────────
def collect_raw_files() -> list[Path]:
    files = []
    for d in RAW_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*.md")))
    return files

# ── OpenAI 호출 ───────────────────────────────────────
def extract_json_with_gpt(text: str, retries: int = 3) -> dict | None:
    """GPT-4o-mini로 텍스트에서 JSON을 추출한다."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            result = json.loads(response.choices[0].message.content)

            # 필수 필드 검증
            if result.get("company") and result.get("position"):
                return result
            else:
                print(f"    [재시도 {attempt+1}] 필수 필드 누락")

        except Exception as e:
            print(f"    [재시도 {attempt+1}] 오류: {e}")
            time.sleep(2 ** attempt)

    return None

# ── Main ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("SBV-LLM GPT-4o-mini 고품질 라벨 생성기")
    print("=" * 50)

    raw_files = collect_raw_files()
    processed = load_checkpoint()
    print(f"원문 파일: {len(raw_files)}건")
    print(f"이미 처리: {len(processed)}건")
    print(f"남은 작업: {len(raw_files) - len(processed)}건\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for i, filepath in enumerate(raw_files):
            file_id = filepath.stem

            if file_id in processed:
                continue

            print(f"[{i+1}/{len(raw_files)}] 처리 중: {filepath.name}")

            # 원문 읽기
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 전처리: 마크다운 주석 제거
            text = "\n".join(
                line for line in content.split("\n")
                if not line.strip().startswith("<!--")
            ).strip()

            # 원티드 특화 노이즈 제거
            if "## 이 포지션을 찾고 계셨나요?" in text:
                text = text.split("## 이 포지션을 찾고 계셨나요?")[0]
            if "본 채용정보는 원티드랩의 동의없이" in text:
                text = text.split("본 채용정보는 원티드랩의 동의없이")[0]
            
            text = text.strip()

            if len(text) < 100:
                print(f"  ⚠️ 텍스트 너무 짧음, 건너뜀")
                fail_count += 1
                processed.add(file_id)
                continue

            # GPT 호출
            result = extract_json_with_gpt(text)

            if result:
                sample = {
                    "instruction": INSTRUCTION_TEMPLATE,
                    "input": text,
                    "output": json.dumps(result, ensure_ascii=False),
                    "source": str(filepath.relative_to(PROJECT_ROOT)),
                }
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_f.flush()
                success_count += 1
                print(f"  ✅ 성공 ({success_count}건)")
            else:
                fail_count += 1
                print(f"  ❌ 실패")

            processed.add(file_id)

            if (success_count + fail_count) % 10 == 0:
                save_checkpoint(processed)

            # GPT-4o-mini는 속도가 빨라 0.2초 정도만 쉬어도 충분합니다.
            time.sleep(0.2)

    save_checkpoint(processed)
    print(f"\n{'='*50}")
    print(f"[완료] 성공: {success_count}건 / 실패: {fail_count}건")
    print(f"[저장] {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
