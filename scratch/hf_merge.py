
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = PROJECT_ROOT / "models" / "sbv-dora-adapter"
MERGED_PATH = PROJECT_ROOT / "models" / "sbv-llm-merged"

print("1. 베이스 모델 로딩 (CPU 메모리 약 18GB 사용)...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    torch_dtype=torch.bfloat16,
    device_map="cpu", # CPU 메모리 사용 (VRAM 부족 방지)
    low_cpu_mem_usage=True
)

print("2. DoRA 어댑터 결합...")
model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

print("3. 모델 병합 중 (이 과정은 시간이 조금 걸릴 수 있습니다)...")
model = model.merge_and_unload()

print("4. 병합된 모델 저장 중...")
model.save_pretrained(str(MERGED_PATH))

print("5. 토크나이저 저장 중...")
tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))
tokenizer.save_pretrained(str(MERGED_PATH))

print(f"\n✅ 병합 성공! 저장 위치: {MERGED_PATH}")
