# ADR: Architecture Decision Records

## ADR-001: Base Model Selection — Qwen 3.5 (9B)

**Status:** Accepted
**Context:** 12GB VRAM에서 JSON 추출 전용 모델을 파인튜닝해야 함. 후보: Qwen 3.5 (9B Dense), Gemma-4-E4B (8B Dense), Llama-4-Scout (17B MoE).
**Decision:** Qwen 3.5 (9B) 선택.
**Rationale:**
- 9B Dense 구조로 한국어 인코딩 효율이 높고 다국어 및 IT 지식 표현력이 우수함.
- Unsloth 공식 지원 확인 → 학습 가속 2x, 메모리 70% 절감으로 12GB VRAM에서 원활한 학습 가능.
- GGUF 변환 및 Ollama 호환이 안정적이어서 경량 로컬 서빙에 최적화됨.
- Base 모델 사용 시 "Base → Expert" 포트폴리오 스토리 차별화.
- **Fallback:** Base 미존재 시 IT(Instruct-Tuned) 버전 사용. IT는 데이터 적어도 수렴 가능하므로 리스크 낮음.

**Rejected Alternatives:**
- Llama-4-Scout: 17B MoE로 GGUF 변환이 복잡하고 Ollama 호환성이 불확실함.
- Gemma-4-E4B: 한국어 공고 텍스트에 대한 어휘 사전(Vocabulary) 효율 및 인코딩 성능이 Qwen 3.5 대비 상대적으로 떨어짐.

---

## ADR-002: DoRA over LoRA

**Status:** Accepted
**Context:** PEFT 방식으로 LoRA vs DoRA 중 선택.
**Decision:** DoRA를 주 학습으로, LoRA를 비교 대조군으로 병행.
**Rationale:**
- DoRA는 가중치 방향/크기를 분리 학습 → 동일 rank에서 LoRA 대비 정확도 향상 (논문 검증)
- Unsloth에서 `use_dora=True` 한 줄로 전환 가능 → 추가 구현 비용 0
- LoRA를 비교군으로 남김으로써 벤치마크 리포트에 정량적 비교 데이터 확보 (포트폴리오 가치 극대화)

---

## ADR-003: Data Source Selection — 원티드 1,000건

**Status:** Accepted
**Context:** 학습용 채용 공고 1,000건을 어디서 수집할 것인가.
**Decision:** 원티드(1,000건) 단일 소스.
**Rationale:**
- 원티드: 국내 개발자 채용 시장의 핵심 플랫폼으로, 텍스트가 풍부하고 개발자 기술 스택명이 다른 플랫폼에 비해 정형화되어 제공되어 정보 추출 모델 학습(Supervised Learning)용 데이터로 최적의 품질을 가짐.
- 1,000건 수집 시 데이터의 다양성 및 분포가 충분하여, 여러 플랫폼(예: 직행 등 애그리게이터)을 병행 수집할 때 발생하는 상이한 HTML 레이아웃 오염 리스크를 차단하고 데이터 정제 엔진의 집중도를 높일 수 있음.
- **법적 리스크:** 포트폴리오 및 비영리 학습 용도로 제한하며, 비공개로 학습을 수행해 크롤링 저작권 및 플랫폼 약관 관련 리스크를 철저히 차단.
- **안전 설정:** robots.txt 준수, 2초 딜레이, User-Agent 명시, 동시 1 요청.

**Rejected Alternatives:**
- 직행: 28개 플랫폼의 공고를 무분별하게 크롤링한 결과물이라 HTML 마크다운 구조의 변동 폭이 너무 커 전처리 노이즈 필터의 정밀도가 떨어지는 한계로 기각.
- 잡코리아/사람인: 크롤링 소송 전례(120억)가 있고, 일반 직무 공고 비율이 높아 개발 스택 추출에 부적합.
- LinkedIn: 강력한 자동 봇 차단 정책으로 실시간 수집 불가.

---

## ADR-004: Evaluation Framework Design

**Status:** Accepted
**Context:** JSON 추출 품질을 어떻게 정량 평가할 것인가.
**Decision:** 5개 지표 + 필드별 비교 규칙.
**Rationale:**

| 지표 | 측정 대상 | 왜 필요한가 |
|------|----------|------------|
| JSON 유효율 | `json.loads()` 성공률 | 기본 동작 보장 |
| 필드별 Exact Match | 7개 필드 개별 정답률 | 실제 추출 정확도 |
| 스키마 준수율 | required 필드 존재 여부 | 다운스트림 호환성 |
| 평균 Latency | 1건당 추론 시간 | 실용성 |
| 비용/건 | API 과금 vs 로컬 | 가성비 입증 |

**필드 비교 규칙 (설계 의도):**
- `tech_stack` 순서 무시: 공고마다 기술 나열 순서가 다르므로 set 비교
- 대소문자 무시: "Python" vs "python" 차이는 무의미
- recall 기준: 정답에 있는 항목이 모두 추출되었는가가 핵심. 추가 추출은 허용 (정보 손실보다 과잉이 나음)
- null 비교: 정답이 null이면 예측도 null이어야 정답 (빈 문자열 ≠ null)

---

## ADR-005: Fallback Strategy — Gemini 2.5 Flash

**Status:** Accepted
**Context:** SBV-LLM이 유효하지 않은 JSON을 출력할 경우 대응.
**Decision:** Retry 2회 → 실패 시 Gemini 2.5 Flash API에 동일 스키마로 폴백.
**Rationale:**
- Gemini의 `response_schema` 기능으로 **동일 JSON 스키마를 서버사이드에서 강제** 가능
- 다운스트림 코드가 SBV 출력이든 Gemini 출력이든 동일하게 처리 가능
- 폴백 비율이 높으면(>5%) 학습 데이터 품질을 재검토해야 한다는 신호

---

## ADR-006: 800건 수집 + 증강 2,000건+ 전략

**Status:** Accepted
**Context:** 학습 데이터 규모 결정. 300건 vs 500건 vs 800건 vs 1,000건.
**Decision:** 원본 800건 수집, 증강으로 2,000건+ 학습 데이터 구축.
**Rationale:**
- Base 모델은 IT 대비 더 많은 데이터 필요 → 500건은 경계선
- 800건 × ~2.5 증강 = 2,000건 → Base 모델이라도 안정적 수렴
- 1,000건 이상 수집은 시간 대비 한계 효용 체감
- 증강 유형: 전체 추출(750) + 부분 추출(750) + 지시어 변형(375) + 엣지(~100)
- **골든 데이터셋 50건**: 학습에 절대 미사용, 평가 전용. Gemini가 생성한 라벨을 수동 검수하여 정답 보장.

---

## ADR-007: Qwen Native Chat Template

**Status:** Accepted
**Context:** 학습 데이터의 instruction-input-output 포맷을 무엇으로 할 것인가.
**Decision:** Qwen 네이티브 chat template (ChatML) 사용.
**Rationale:**
- Qwen은 `<|im_start|>user ... <|im_end|>` 형식의 자체 토큰 보유
- Unsloth가 Qwen chat template을 자동 적용 (get_chat_template 사용) → 수동 토큰 삽입 불필요
- Alpaca 포맷 대비 모델 네이티브 포맷이 성능상 유리 (토큰 임베딩이 이미 학습됨)
- `SFTTrainer`의 `formatting_func`으로 JSONL → chat template 자동 변환
