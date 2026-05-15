"""
08_export_gguf.py — GGUF 변환 + Ollama 등록

DoRA 어댑터 → Merge → GGUF(q4_k_m) → Ollama 모델 생성.

Usage:
    python scripts/08_export_gguf.py
"""

import subprocess
import yaml
import os
from pathlib import Path
from unsloth import FastLanguageModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
ADAPTER_DIR = PROJECT_ROOT / "models" / "sbv-llm-merged" # 병합된 모델 경로로 변경
GGUF_DIR = PROJECT_ROOT / "models"
MODELFILE_PATH = PROJECT_ROOT / "Modelfile"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def main():
    print("=" * 50)
    print("SBV-LLM GGUF 변환 + Ollama 등록")
    print("=" * 50)

    # 1. 병합된 모델 로드 (GPU VRAM 대신 CPU RAM 사용)
    print("모델 로드 중 (CPU)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=256,
        dtype=None,
        load_in_4bit=False,
        device_map="cpu", # VRAM OOM 원천 차단
    )

    # 2. GGUF 변환
    quant = cfg["export"]["gguf_quantization"]
    output_name = f"sbv-llm-{quant.replace('_', '-')}"
    print(f"\nGGUF 변환: {quant}")

    model.save_pretrained_gguf(
        str(GGUF_DIR / output_name),
        tokenizer,
        quantization_method=quant,
    )
    print(f"✅ GGUF 저장: {GGUF_DIR / output_name}")

    # 3. Modelfile 생성
    gguf_path = list(GGUF_DIR.glob(f"{output_name}*.gguf"))
    if not gguf_path:
        print("❌ GGUF 파일을 찾을 수 없습니다.")
        return

    modelfile_content = f"""FROM {gguf_path[0]}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.2
PARAMETER num_predict 4096
PARAMETER num_ctx 16384

SYSTEM \"\"\"당신은 채용 공고에서 구조화된 정보를 추출하는 전문가입니다.
주어진 채용 공고 텍스트에서 다음 필드를 추출하여 오직 유효한 JSON 포맷으로만 출력하세요. 
정보가 텍스트에 명시되지 않은 경우, 해당 필드의 값은 null 또는 빈 배열([])로 설정하세요. 절대 정보를 지어내지 마세요.

JSON Schema:
{{
  "company": "회사명 (string)",
  "position": "채용 포지션 명 (string)",
  "tech_stack": ["기술 스택 리스트 (array of strings)"],
  "experience": "요구 경력 (string)",
  "location": "근무지 주소 (string)",
  "salary_range": "연봉 범위 (string or null)",
  "employment_type": "고용 형태 (string or null)",
  "main_tasks": ["주요 업무 리스트 (array of strings)"],
  "requirements": ["자격 요건 리스트 (array of strings)"],
  "preferred_qualifications": ["우대 사항 리스트 (array of strings)"],
  "benefits": ["복지 혜택 리스트 (array of strings)"]
}}

주의사항:
1. tech_stack은 본문에 명시된 기술 키워드(예: Python, AWS, React)만 추출하세요.
2. main_tasks, requirements, preferred_qualifications, benefits는 본문의 내용을 바탕으로 핵심 문구만 리스트로 만드세요.
3. 본문에 없는 기술이나 정보는 절대 창작하지 마세요 (Hallucination 금지).
4. 모든 출력은 반드시 유효한 JSON 형태여야 하며, 다른 설명 문구는 포함하지 마세요.\"\"\"
"""
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"✅ Modelfile 생성: {MODELFILE_PATH}")

    # 4. Ollama 등록
    print("\nOllama 모델 등록 중...")
    try:
        subprocess.run(["ollama", "create", "sbv-llm", "-f", str(MODELFILE_PATH)], check=True)
        print("✅ Ollama 등록 완료: sbv-llm")

        # 테스트
        print("\n테스트 추론:")
        result = subprocess.run(
            ["ollama", "run", "sbv-llm", "테스트: 카카오에서 백엔드 개발자를 모집합니다. Python, Django 경험 3년 이상."],
            capture_output=True, text=True, timeout=60,
        )
        print(result.stdout)
    except FileNotFoundError:
        print("⚠️ Ollama가 설치되어 있지 않습니다. 수동으로 등록하세요:")
        print(f"   ollama create sbv-llm -f {MODELFILE_PATH}")
    except subprocess.TimeoutExpired:
        print("⚠️ 테스트 추론 타임아웃")


if __name__ == "__main__":
    main()
