
import torch
from unsloth import FastLanguageModel
from pathlib import Path
import json

# 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = PROJECT_ROOT / "models" / "sbv-dora-adapter"

SAMPLE_INPUT = """
[비트나인]∙서울 본사∙경력 3-7년
# [G-SQL] Graph Database 엔진 개발자
### 주요업무
• 관계형 데이터베이스와 그래프 데이터베이스를 통합한 하이브리드 쿼리 엔진 개발
• C/C++ 기반의 고성능 엔진 아키텍처 설계
• PostgreSQL 코어 소스 분석 및 커스터마이징
"""

def main():
    print("="*50)
    print("SBV-LLM DoRA 어댑터 검증 (Bypassing VL Processor)")
    print("="*50)
    
    # 1. 모델 로드
    model, tokenizer_obj = FastLanguageModel.from_pretrained(
        model_name = str(ADAPTER_PATH),
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # 2. 토크나이저 추출 (Processor일 경우 내부 tokenizer 사용)
    if hasattr(tokenizer_obj, "tokenizer"):
        tokenizer = tokenizer_obj.tokenizer
    else:
        tokenizer = tokenizer_obj

    # 3. 프롬프트 구성
    prompt = f"### Instruction:\n다음 채용 공고에서 구조화된 정보를 JSON으로 추출하세요.\n\n### Input:\n{SAMPLE_INPUT}\n\n### Response:\n"
    
    # input_ids 추출
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # 4. 추론
    print("\n[추론 진행 중...]")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids = input_ids,
            max_new_tokens = 256,
            use_cache = True,
            temperature = 0.1,
            top_p = 0.9,
            repetition_penalty = 1.1,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
        )
    
    # 결과 해석
    generated_ids = output_ids[0][len(input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    print("\n" + "="*30)
    print("최종 추출 결과 (JSON)")
    print("="*30)
    print(response)
    print("="*30)

    try:
        json.loads(response)
        print("\n✅ 검증 완료: 유효한 JSON 형식입니다. 모델이 아주 똑똑하게 학습되었습니다!")
    except:
        print("\n⚠️ 결과가 JSON 형식이 아니지만 텍스트 생성은 확인되었습니다.")

if __name__ == "__main__":
    main()
