"""
05_train_dora.py — DoRA 파인튜닝

최신 LLM(Qwen 3.5 등)을 DoRA(Weight-Decomposed LoRA)로 파인튜닝.
configs/train_config.yaml의 설정을 사용.

Usage:
    python scripts/05_train_dora.py

Requirements:
    pip install unsloth peft bitsandbytes transformers datasets trl
"""

import json
import os
# PyTorch Dynamo 컴파일러 충돌 방지용 (최상단 배치)
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = "-dynamo"

import yaml
from pathlib import Path
from datasets import load_dataset, DatasetDict
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 128
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 128

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train_augmented.jsonl"
VAL_PATH = PROJECT_ROOT / "data" / "splits" / "val.jsonl"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def formatting_func(examples, tokenizer):
    """JSONL → Chat Template 변환 (tokenizer.apply_chat_template 사용)."""
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        conversations = [
            {"role": "user", "content": f"{inst}\n\n{inp}"},
            {"role": "assistant", "content": out},
        ]
        # 모델별 최적화된 템플릿 적용
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def main():
    print("=" * 50)
    print("SBV-LLM DoRA 파인튜닝")
    print("=" * 50)

    # 1. 모델 로드
    base_model = cfg["model"]["base_model"]
    print(f"모델 로드: {base_model}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=cfg["training"]["max_seq_length"], # 입력 토큰 최대 길이 (이거 넘어가면 짤림)
            dtype=None,                                       # None으로 두면 VRAM 상황 맞춰서 bfloat16/float16 자동 스니핑함
            load_in_4bit=cfg["quantization"]["load_in_4bit"], # 4bit 양자화 켜기 (12GB VRAM에서 8B 모델 돌리려면 필수)
            device_map={"": 0},                               # Accelerate의 보수적 메모리 계산을 무시하고 강제로 GPU 0에 할당
        )
    except Exception:
        # Base 모델 다운로드 실패하거나 없으면 Instruct(IT) 모델로 변경
        base_model = cfg["model"]["fallback_model"]
        print(f"[폴백] Base 모델 없음 → {base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=cfg["training"]["max_seq_length"],
            dtype=None,
            load_in_4bit=cfg["quantization"]["load_in_4bit"],
            device_map={"": 0},

        )

    # Base 모델용 챗 템플릿 강제 주입 (Instruct 모델이 아닐 경우 템플릿이 없어서 에러나는 것 방지)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    # 2. LoRA/DoRA 어댑터 적용
    lora_cfg = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],                            # rank값. 크면 똑똑해지지만 VRAM 많이 먹음 (보통 16이나 32 줌)
        lora_alpha=lora_cfg["lora_alpha"],          # alpha 스케일링. 관습적으로 r의 2배 세팅함
        lora_dropout=lora_cfg["lora_dropout"],      # unsloth 최적화 제대로 태우려면 0으로 둬야 연산 빠름
        target_modules=lora_cfg["target_modules"],  # Q,K,V,O 등 타겟 모듈 전체에 다 붙여야 성능 잘 나옴
        use_dora=True,                              # DoRA 활성화. 그냥 LoRA보다 추출 정밀도가 훨씬 높음
        use_gradient_checkpointing=True,            # "unsloth" 전용 대신 파이토치 표준 최적화(True) 사용. VRAM은 여전히 절약되면서 에러 안 남
        random_state=3407,                          # AI 업계의 '행운의 숫자'. 재현성을 위해 난수 시드 고정
    )

    # 3. 데이터셋 로드
    # 두 데이터셋의 컬럼 구조가 다를 수 있으므로(예: source 유무) 따로 불러서 불필요한 컬럼을 제거하고 합침
    train_ds = load_dataset("json", data_files=str(TRAIN_PATH), split="train")
    val_ds = load_dataset("json", data_files=str(VAL_PATH), split="train")
    
    cols_to_remove = [c for c in val_ds.column_names if c not in train_ds.column_names]
    if cols_to_remove:
        val_ds = val_ds.remove_columns(cols_to_remove)
        
    dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })
    dataset = dataset.map(
        formatting_func,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )

    # 4. 학습 세팅
    tcfg = cfg["training"]
    training_args = SFTConfig(
        output_dir=str(PROJECT_ROOT / "models" / "sbv-dora-adapter"),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"], # GPU 1대당 배치 사이즈. OOM(메모리 터짐) 나면 무조건 이거부터 줄여야 됨
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"], # 배치 작게 잡은 대신 여러번 모아서 업데이트 (실질적 배치사이즈 키움)
        num_train_epochs=tcfg["num_train_epochs"],                       # 전체 데이터 몇 바퀴 돌릴건지 (보통 3바퀴 돌리면 적당함)
        learning_rate=tcfg["learning_rate"],                             # 학습률. 2e-4가 국룰이긴 한데 loss 너무 튀면 줄여야 함
        weight_decay=tcfg["weight_decay"],                               # 가중치 너무 튀는거 눌러주는 용도
        warmup_ratio=tcfg["warmup_ratio"],                               # 초반에 LR 서서히 올리는 비율 (보통 0.03~0.1 사이 줌)
        lr_scheduler_type=tcfg["lr_scheduler_type"],                     # 학습률 줄이는 방식 (요즘은 거의 cosine 씀)
        bf16=tcfg["bf16"],                                               # Ampere 이상(RTX 30, 40, 50 시리즈) 글카면 fp16 말고 무조건 bf16 켜는게 연산 이득
        logging_steps=tcfg["logging_steps"],                             # 몇 스텝마다 로그 찍을건지
        save_strategy=tcfg["save_strategy"],                             # epoch 끝날때마다 체크포인트 딸건지 설정
        optim=tcfg["optimizer"],                                         # 8bit optimizer 쓰면 옵티마이저가 먹는 VRAM 엄청 아껴짐 (paged_adamw_8bit 추천)
        max_seq_length=tcfg["max_seq_length"],                           # 모델이 한 번에 처리할 최대 토큰 수 (VRAM 용량에 따라 조절)
        dataset_text_field="text",                                       # 데이터셋 텍스트 컬럼명 매핑
        gradient_checkpointing=True,                                     # "unsloth" 전용 컴파일러 끄고 안정적인 표준 체크포인팅 적용
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )

    print("\n학습 시작...")
    trainer.train()

    # 5. 어댑터 저장
    output_dir = PROJECT_ROOT / "models" / "sbv-dora-adapter"
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n[완료] 어댑터 저장: {output_dir}")


if __name__ == "__main__":
    main()
