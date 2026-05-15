
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/sbv-llm-merged",
    load_in_4bit=False,
    device_map="cpu", # Force CPU to avoid VRAM OOM
    torch_dtype=torch.bfloat16, # Same as what we saved
)

print("모델 로드 완료. GGUF 변환 시작...")
model.save_pretrained_gguf("models/sbv-llm-q4_k_m", tokenizer, quantization_method="q4_k_m")
print("GGUF 변환 완료!")
