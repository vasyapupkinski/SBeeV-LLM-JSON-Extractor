import json
from pathlib import Path

FILE_PATH = Path("data/labeled/labeled_dataset_v2.jsonl")

def check_v2_dataset():
    if not FILE_PATH.exists():
        print("파일이 존재하지 않습니다.")
        return

    missing_deadline_count = 0
    total_lines = 0
    duplicates = set()
    duplicate_count = 0
    
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total_lines += 1
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                output = json.loads(data["output"])
                
                # 마감일 필드 확인
                if "deadline" not in output:
                    missing_deadline_count += 1
                
                # 회사명+포지션으로 중복 확인
                comp_pos = f"{output.get('company')}_{output.get('position')}"
                if comp_pos in duplicates:
                    duplicate_count += 1
                duplicates.add(comp_pos)
                
            except:
                pass
                
    print(f"--- 2차 정밀 검수 결과 ---")
    print(f"전체 줄 수: {total_lines}")
    print(f"마감일(deadline) 필드 누락: {missing_deadline_count}건")
    print(f"실제 데이터 중복(회사+포지션 동일): {duplicate_count}건")
    print(f"-------------------------")

if __name__ == "__main__":
    check_v2_dataset()
