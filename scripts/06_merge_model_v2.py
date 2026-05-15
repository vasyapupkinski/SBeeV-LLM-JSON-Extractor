import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 이번에 학습시킨 V2 어댑터 경로
ADAPTER_PATH = PROJECT_ROOT / "models" / "sbv-dora-adapter-v2"
# 저장될 병합 모델 경로
MERGED_PATH = PROJECT_ROOT / "models" / "sbv-llm-merged-v2"

def main():
    print("=" * 50)
    print("V2 모델 병합 시작 (CPU Safe Mode)")
    print("=" * 50)

    # 1. 베이스 모델 로딩
    print("1. 베이스 모델 로딩 (CPU 사용)...")
    base_model_id = "Qwen/Qwen3.5-9B"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
        low_cpu_mem_usage=True
    )

    # 2. V2 DoRA 어댑터 결합
    print("2. V2 DoRA 어댑터 결합...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    # 3. 모델 병합
    print("3. 모델 병합 중 (시간이 조금 걸립니다)...")
    model = model.merge_and_unload()

    # 4. 병합된 모델 저장
    print("4. 병합된 모델 저장 중...")
    MERGED_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_PATH))

    # 5. 토크나이저 저장
    print("5. 토크나이저 저장 중...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
    tokenizer.save_pretrained(str(MERGED_PATH))

    print(f"\n✅ V2 병합 성공! 저장 위치: {MERGED_PATH}")

if __name__ == "__main__":
    main()
