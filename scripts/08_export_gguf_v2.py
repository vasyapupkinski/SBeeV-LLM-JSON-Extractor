import subprocess
import yaml
import os
from pathlib import Path
from unsloth import FastLanguageModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config_v2.yaml"
# 병합된 V2 모델 경로 (06_merge_model_v2.py 결과물)
ADAPTER_DIR = PROJECT_ROOT / "models" / "sbv-llm-merged-v2"
# 결과물이 저장될 폴더 (Unsloth는 입력 모델명 뒤에 _gguf를 붙이는 경향이 있습니다)
GGUF_DIR = PROJECT_ROOT / "models" / "sbv-llm-merged-v2_gguf"
MODELFILE_PATH = PROJECT_ROOT / "Modelfile_v2"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def main():
    print("=" * 50)
    print("SBV-LLM V2 (Deadline) GGUF 변환 + Ollama 등록")
    print("=" * 50)

    if not ADAPTER_DIR.exists():
        print(f"❌ 어댑터 폴더를 찾을 수 없습니다: {ADAPTER_DIR}")
        return

    # 1. 모델 및 어댑터 로드 (GPU VRAM 대신 CPU RAM 사용)
    print(f"어댑터 로드 중 (CPU): {ADAPTER_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=256, # 변환 시에는 길 필요 없음
        dtype=None,
        load_in_4bit=False, # CPU 로드 시에는 4bit 필요 없음
        device_map="cpu",   # VRAM 터짐 방지
    )

    # 2. GGUF 변환 및 병합 저장
    quant = cfg["export"]["gguf_quantization"]
    print(f"\nGGUF 변환 시작 (방식: {quant})...")

    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(GGUF_DIR),
        tokenizer,
        quantization_method=quant,
    )
    print(f"✅ GGUF 변환 및 병합 완료: {GGUF_DIR}")

    # 3. Ollama 등록
    # GGUF 파일 찾기
    gguf_path = list(GGUF_DIR.glob("*.gguf"))
    if not gguf_path:
        print("❌ 변환된 GGUF 파일을 찾을 수 없습니다.")
        return
    
    actual_gguf = gguf_path[0]
    print(f"✅ 찾은 GGUF 파일: {actual_gguf}")

    # Modelfile_v2 내용 업데이트 (실제 GGUF 경로 주입)
    system_prompt = """당신은 채용 공고에서 구조화된 정보를 추출하는 전문가입니다.
주어진 채용 공고 텍스트에서 다음 필드를 추출하여 오직 유효한 JSON 포맷으로만 출력하세요. 
정보가 텍스트에 명시되지 않은 경우, 해당 필드의 값은 null 또는 빈 배열([])로 설정하세요. 절대 정보를 지어내지 마세요.

JSON Schema:
{
  "company": "회사명 (string)",
  "position": "채용 포지션 명 (string)",
  "tech_stack": ["기술 스택 리스트 (array of strings)"],
  "experience": "요구 경력 (string)",
  "location": "근무지 주소 (string)",
  "deadline": "마감일 (string, 예: 2026-12-31 또는 상시채용)",
  "salary_range": "연봉 범위 (string or null)",
  "employment_type": "고용 형태 (string or null)"
}

주의사항:
1. tech_stack은 본문에 명시된 기술 키워드만 추출하세요.
2. 마감일(deadline)은 날짜 형식이나 '상시채용' 등의 문구를 그대로 추출하세요.
3. 모든 출력은 반드시 유효한 JSON 형태여야 합니다."""

    modelfile_content = f"""FROM {actual_gguf}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.0
PARAMETER num_ctx 16384

SYSTEM \"\"\"{system_prompt}\"\"\"
"""
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"✅ Modelfile_v2 업데이트 완료: {MODELFILE_PATH}")

    print("\nOllama 모델 등록 중 (sbv-llm-v2)...")
    try:
        subprocess.run(["ollama", "create", "sbv-llm-v2", "-f", str(MODELFILE_PATH)], check=True)
        print("✅ Ollama 등록 완료: sbv-llm-v2")
    except Exception as e:
        print(f"⚠️ Ollama 등록 실패: {e}")
        print(f"   수동 등록 명령어: ollama create sbv-llm-v2 -f {MODELFILE_PATH}")

if __name__ == "__main__":
    main()
