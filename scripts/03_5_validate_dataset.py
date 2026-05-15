import json
import os
from collections import Counter

def validate_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    total_count = 0
    error_count = 0
    missing_fields = Counter()
    tech_stack_empty = 0
    samples = []

    print(f"{file_path} 데이터 검증 시작...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_count += 1
            try:
                data = json.loads(line)
                
                # 출력 결과물(output) 파싱 시도
                output_str = data.get('output', '{}')
                try:
                    output = json.loads(output_str)
                except json.JSONDecodeError:
                    error_count += 1
                    print(f"⚠️ {line_num}행 output JSON 파싱 오류")
                    continue
                
                # 필수 필드 체크
                required = ['company', 'position', 'experience', 'location']
                for field in required:
                    val = output.get(field)
                    if val is None or str(val).strip() == "":
                        missing_fields[field] += 1
                
                # 기술 스택 비어있는지 확인
                tech_stack = output.get('tech_stack', [])
                if not tech_stack or len(tech_stack) == 0:
                    tech_stack_empty += 1
                
                # 샘플 저장 (앞부분 5개)
                if len(samples) < 5:
                    samples.append(output)

            except Exception as e:
                error_count += 1
                print(f"⚠️ {line_num}행 오류: {str(e)}")

    # 결과 요약
    print("\n" + "="*50)
    print(f" 총 데이터 개수: {total_count}")
    print(f" JSON 파싱 에러: {error_count}")
    print(f" 기술 스택 미추출(Empty): {tech_stack_empty} ({tech_stack_empty/total_count*100:.1f}%)")
    print("-" * 50)
    print(" 필수 필드 누락(Null 또는 빈 문자열) 통계:")
    if not missing_fields:
        print("  - 누락된 필드 없음 (완벽합니다!)")
    else:
        for field, count in missing_fields.items():
            print(f"  - {field}: {count}건 ({count/total_count*100:.1f}%)")
    
    print("\n🔍 상위 5개 샘플 미리보기:")
    for idx, s in enumerate(samples, 1):
        print(f"  {idx}. {s.get('company', 'N/A')} | {s.get('position', 'N/A')} | {s.get('tech_stack', [])}")
    print("="*50)

if __name__ == "__main__":
    # 현재 디렉토리 기준 경로 처리 (최상위 디렉토리에서 실행한다고 가정)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "labeled", "labeled_dataset.jsonl")
    
    validate_dataset(dataset_path)
