# PRD: SBV-LLM JSON Extraction Engine

## Problem
채용 공고 등 비정형 텍스트에서 구조화 JSON을 추출할 때, 상용 LLM(Gemini, GPT)은 비용이 발생하고, 소형 오픈소스 모델은 JSON 형식을 자주 깨뜨린다. 12GB VRAM 로컬 환경에서 비용 0원으로 안정적인 JSON 추출이 필요하다.

## Goal
Qwen 3.5 (9B)를 DoRA 파인튜닝하여 채용 공고 → 구조화 JSON을 95%+ 유효율로 추출하는 특화 엔진 구축.

## Non-Goals
- 범용 대화/요약 기능 (Devoffs의 범용 LLM이 담당)
- 실시간 크롤링 서비스 (수집은 오프라인 배치)
- 웹 UI (API 서빙만, UI는 Devoffs 프로젝트)

## Output Schema
```json
{
  "company": "string (required)",
  "position": "string (required)",
  "tech_stack": ["string"] | [],
  "experience": "string | null",
  "location": "string | null",
  "salary_range": "string | null",
  "employment_type": "string | null"
}
```
**Nullable 정책:** 공고에 없는 필드 → `null`. `tech_stack` 없으면 `[]`. `company`, `position`은 항상 필수.

## Data Sources
| 소스 | 수집량 | 특성 |
|------|:------:|------|
| 원티드 (wanted.co.kr) | 1,000건 | 개발자 채용 주력, 상세 공고 |

원본 1,000건 → 증강(지시어 변형 + 부분 추출 + 엣지 케이스) → 학습 데이터 3,000건+


## Success Metrics
| 지표 | 목표 |
|------|:----:|
| JSON 유효율 | ≥95% |
| 필드별 Exact Match | ≥85% |
| 스키마 준수율 | ≥98% |
| 평균 Latency (Ollama) | ≤1.5s |
| 비용/건 | 0원 |

## Constraints
- **VRAM:** 12GB (RTX 5070) — QLoRA 4-bit 필수
- **모델:** Qwen 3.5 (9B) (Base 우선, 없으면 IT)
- **법적:** 포트폴리오 용도, 크롤링 데이터 비공개, robots.txt 준수
- **Fallback:** SBV-LLM 실패 시 Gemini 2.5 Flash API로 동일 스키마 폴백

## Downstream Consumer
Devoffs (채용 플랫폼)가 SBV-LLM의 JSON 출력을 DB에 저장 → 검색/필터링에 활용. 원문은 별도 저장하며, 요약은 범용 LLM이 담당.
