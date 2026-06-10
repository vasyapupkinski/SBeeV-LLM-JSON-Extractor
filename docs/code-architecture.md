# Code Architecture: SBV-LLM

## Project Structure

```
SBeeV-LLM-JSON-Extractor/
├── docs/                          # 이 문서들
├── scripts/
│   ├── 01_crawl_wanted.py         # 원티드 크롤러
│   ├── 03_generate_labels_v2.py   # Gemini 정답 생성
│   ├── 04_augment_dataset_v2.py   # 데이터 증강
│   ├── 05_train_dora_v2.py        # DoRA 파인튜닝 (주 학습)
│   ├── 06_merge_model_v2.py       # 모델 병합
│   ├── 07_evaluate_v2.py          # 벤치마크 평가
│   ├── 08_export_gguf_v2.py       # GGUF 변환 + Ollama 등록
│   └── 09_bulk_inference_v2.py    # 벌크 추론 및 정보 추출
├── data/
│   ├── raw/wanted/*.md            # 원티드 크롤링 원문
│   ├── labeled/*.jsonl            # Gemini 생성 정답
│   ├── splits_v2/{train,val}.jsonl# 학습/검증 분리
│   └── golden/golden_50.jsonl     # 평가 전용 (학습 미사용)
├── models/
│   ├── sbv-dora-adapter/          # DoRA 어댑터 (~50MB)
│   └── sbv-llm-q4_k_m.gguf        # 배포 모델
├── results_v2/                    # 벤치마크 결과
├── configs/
│   ├── schema.json                # JSON 추출 스키마
│   ├── crawl_config.yaml          # 크롤러 설정
│   └── train_config_v2.yaml       # 학습 하이퍼파라미터
├── Modelfile_v2                   # Ollama 모델 정의
└── requirements.txt
```

## Module Responsibilities

### scripts/01: Crawlers
- **입력:** 사이트 URL + 크롤 설정
- **출력:** `data/raw/wanted/{id}.md`
- **의존:** `crawl4ai`
- **주의:** robots.txt 준수, 2초 딜레이, User-Agent 명시, 체크포인트 저장(중단 재개 가능)

### scripts/03: Label Generator
- **입력:** `data/raw/*.md`
- **출력:** `data/labeled/labeled_dataset.jsonl`
- **의존:** `google-generativeai`
- **형식:** `{"instruction": "...", "input": "원문", "output": "{JSON}"}`
- **주의:** Gemini의 `response_schema`로 스키마 강제. 실패 건 로그 + 재시도.

### scripts/04: Augmenter
- **입력:** `data/labeled/*.jsonl`
- **출력:** `data/augmented/`, `data/golden/`, `data/splits/`
- **증강 유형:** 전체 추출 / 부분 추출 / 지시어 변형 / 엣지 케이스
- **주의:** 골든 50건을 먼저 분리한 뒤 나머지를 증강. train:val = 90:10.

### scripts/05-06: Trainers
- **입력:** `data/splits/train.jsonl`, `configs/train_config.yaml`
- **출력:** `models/sbv-{dora,lora}-adapter/`
- **의존:** `unsloth`, `peft`, `bitsandbytes`, `transformers`
- **핵심 설정:**
  ```yaml
  base_model: Qwen/Qwen3.5-9B  # 또는 Qwen3.5-9B-Instruct
  quant: nf4, double_quant
  lora_r: 16, lora_alpha: 32
  use_dora: true  # 05에서 true, 06에서 false
  seq_length: 2048
  batch: 1, grad_accum: 4
  epochs: 3, lr: 2e-4
  ```

### scripts/07: Evaluator
- **입력:** `data/golden/golden_50.jsonl` + 3개 모델(DoRA, LoRA, Gemini)
- **출력:** `results/*.json`, `results/benchmark_report.md`
- **비교 로직:** 대소문자 무시, 순서 무시, recall 기준 (see ADR-004)

### scripts/08: Exporter
- **입력:** `models/sbv-dora-adapter/`
- **출력:** `models/sbv-llm-q4_k_m.gguf` + Ollama 등록
- **방법:** `model.save_pretrained_gguf()` (Unsloth 내장)

## Key Dependencies
```
unsloth          # 학습 가속 + GGUF 변환
peft             # LoRA/DoRA 어댑터
bitsandbytes     # 4-bit 양자화
transformers     # 모델 로드
datasets         # 데이터셋 처리
crawl4ai         # 웹 스크래핑
google-generativeai  # Gemini API
ollama           # 로컬 LLM 서빙
jsonschema       # 스키마 검증
```

## .gitignore Policy
```
data/raw/        # 크롤링 원문 (저작권)
data/labeled/    # 파생 데이터
data/augmented/  # 파생 데이터
models/*.gguf    # 대용량 바이너리
.env             # API 키
```
공개 대상: `scripts/`, `configs/`, `docs/`, `results/`, `Modelfile`
