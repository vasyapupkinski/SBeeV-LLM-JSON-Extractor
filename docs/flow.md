# Flow: SBV-LLM Pipeline

## End-to-End Flow

```mermaid
flowchart TD
    subgraph "Phase 1: Data Collection"
        A[원티드] -->|Crawl4AI| B["data/raw/ (1,000 .md files)"]
        B -->|품질 필터링| C["유효 원문 950건+"]
    end

    subgraph "Phase 2: Label Generation"
        C -->|GPT-4o-mini| D["data/labeled/ (input-output pairs)"]
        D -->|수동 검수 50건| E["data/golden/ (평가 전용)"]
    end

    subgraph "Phase 3: Augmentation"
        D -->|전체 추출 950건| F[증강 데이터]
        D -->|부분 추출 950건| F
        D -->|지시어 변형 475건| F
        G[엣지 케이스 ~125건] --> F
        F --> H["data/splits_v2/train.jsonl (2,700건)"]
        F --> I["data/splits_v2/val.jsonl (300건)"]
    end

    subgraph "Phase 4: Training"
        H --> J["Unsloth + QLoRA + DoRA"]
        I --> J
        J --> K["models/sbv-dora-adapter/"]
    end

    subgraph "Phase 5: Export & Deploy"
        K -->|save_pretrained_gguf| M["sbv-llm-q4_k_m.gguf"]
        M -->|ollama create| N[Ollama sbv-llm-v2]
    end

    subgraph "Phase 6: Evaluation"
        E --> P[벤치마크 평가]
        N --> P
        Q[GPT-4o-mini] --> P
        P --> R["results_v2/benchmark_report.md"]
    end
```

## Inference Flow (Runtime)

```mermaid
flowchart LR
    A[비정형 텍스트] --> B{SBV-LLM via Ollama}
    B -->|json.loads 성공| C[구조화 JSON]
    B -->|json.loads 실패| D{Retry x2}
    D -->|성공| C
    D -->|실패| E["GPT-4o-mini (Fallback)"]
    E -->|response_schema 강제| C
```

## Augmentation Strategy

| 유형 | 변환 | 입력 → 출력 | 건수 |
|------|------|------------|:----:|
| 전체 추출 | 원본 그대로 | 원문 → 전체 JSON | 950 |
| 부분 추출 | 지시어 변경 | 원문 → `{tech_stack, experience}` only | 950 |
| 지시어 변형 | 프롬프트 3종 | 동일 원문, 다른 지시어 → 동일 JSON | 475 |
| 엣지 케이스 | null 다수 | 비개발/불완전 공고 → null 필드 | ~125 |

## Key Invariants
- 골든 50건은 **절대** 학습 데이터에 포함되지 않음
- 모든 학습 데이터는 `json.loads()`로 파싱 가능해야 함
- 크롤링 요청 간격 ≥ 2초, 동시 요청 1개
