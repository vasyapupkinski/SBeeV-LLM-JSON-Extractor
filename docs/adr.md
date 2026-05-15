# ADR: Architecture Decision Records

## ADR-001: Base Model Selection — Gemma-4-E4B

**Status:** Accepted
**Context:** 12GB VRAM에서 JSON 추출 전용 모델을 파인튜닝해야 함. 후보: Gemma-4-E4B (8B Dense), Llama-4-Scout (17B MoE), Qwen3-8B.
**Decision:** Gemma-4-E4B 선택.
**Rationale:**
- Unsloth 공식 지원 확인 → 학습 가속 2x, 메모리 70% 절감
- 8B Dense 구조 → MoE 대비 GGUF 변환 단순, Ollama 호환 안정적
- Base 모델 사용 시 "Base → Expert" 포트폴리오 스토리 차별화
- **Fallback:** Base 미존재 시 IT(Instruct-Tuned) 버전 사용. IT는 데이터 적어도 수렴 가능하므로 리스크 낮음.

**Rejected Alternatives:**
- Llama-4-Scout: 17B MoE, GGUF 변환 복잡, Ollama 호환 불확실
- Qwen3: Unsloth 지원 미확인 시점

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

## ADR-003: Dual Data Source — 원티드 + 직행

**Status:** Accepted
**Context:** 학습용 채용 공고 800건을 어디서 수집할 것인가.
**Decision:** 원티드(400건) + 직행(400건).
**Rationale:**
- 원티드: 개발자 채용 주력, 상세 공고, API 제공
- 직행: 28개 플랫폼 통합 애그리게이터, IT 공고 4,000건+, 데이터 다양성 확보
- 두 소스 병행 → 특정 사이트의 공고 형식에 오버피팅 방지
- **법적 리스크:** 포트폴리오 용도 + 500건 미만/소스 + 비공개 → 실질적 리스크 제로
- **안전 설정:** robots.txt 준수, 2초 딜레이, User-Agent 명시, 동시 1 요청

**Rejected Alternatives:**
- 잡코리아/사람인: 크롤링 소송 전례(120억), 리스크 불필요
- LinkedIn: 강력한 봇 차단
- 워크넷 API: 합법이지만 개발자 공고 비율 낮음

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

## ADR-007: Gemma 4 Native Chat Template

**Status:** Accepted
**Context:** 학습 데이터의 instruction-input-output 포맷을 무엇으로 할 것인가.
**Decision:** Gemma 4 네이티브 chat template 사용.
**Rationale:**
- Gemma 4는 `<bos><|turn>user ... <turn|><|turn>model ... <turn|>` 형식의 자체 토큰 보유
- Unsloth가 Gemma 4 chat template을 자동 적용 → 수동 토큰 삽입 불필요
- Alpaca 포맷 대비 모델 네이티브 포맷이 성능상 유리 (토큰 임베딩이 이미 학습됨)
- `SFTTrainer`의 `formatting_func`으로 JSONL → chat template 자동 변환
