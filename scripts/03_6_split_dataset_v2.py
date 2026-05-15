"""
03_6_split_dataset.py

라벨링된 데이터셋을 Train, Val, Test (8:1:1 비율)로 분할합니다.
사용자의 제안에 따라 '기술 스택(tech_stack)'의 유무를 기준으로
층화추출(Stratified Split)을 적용하여 클래스 불균형을 방지합니다.
"""

import json
from pathlib import Path
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("scikit-learn이 설치되어 있지 않습니다. 아래 명령어로 설치해주세요:")
    print("pip install scikit-learn")
    exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# V2용 경로 설정
INPUT_FILE = PROJECT_ROOT / "data" / "labeled" / "labeled_dataset_v2.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits_v2"

def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} 파일이 없습니다.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("데이터 로딩 중...")
    data = []
    strata = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data.append(item)
            
            # 기술 스택 추출
            try:
                output_json = json.loads(item["output"])
                tech_stack = output_json.get("tech_stack", [])
                
                # 기술 스택이 비어있으면 1, 있으면 0으로 분류 (Stratify 기준)
                if not tech_stack:
                    strata.append(1)
                else:
                    strata.append(0)
            except json.JSONDecodeError:
                # 파싱 에러가 있는 경우 기본값으로 처리
                strata.append(0)

    total_count = len(data)
    empty_stack_count = sum(strata)
    
    print(f"총 데이터 수: {total_count}")
    print(f"기술 스택 없음 비율: {empty_stack_count / total_count * 100:.1f}%")

    # 1. 80% Train, 20% Temp(Val+Test) 분할
    train_data, temp_data, train_strata, temp_strata = train_test_split(
        data, strata, test_size=0.2, stratify=strata, random_state=42
    )

    # 2. Temp를 다시 반으로 나누어 Val 10%, Test 10%
    val_data, test_data, val_strata, test_strata = train_test_split(
        temp_data, temp_strata, test_size=0.5, stratify=temp_strata, random_state=42
    )

    # 저장 함수
    def save_jsonl(file_path, dataset):
        with open(file_path, "w", encoding="utf-8") as f:
            for d in dataset:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("-" * 50)
    print("분할 결과 (기술 스택 없는 데이터 비율 확인):")
    
    for name, dataset, strt in [("Train", train_data, train_strata), 
                                ("Val", val_data, val_strata), 
                                ("Test", test_data, test_strata)]:
        ratio = sum(strt) / len(dataset) * 100 if len(dataset) > 0 else 0
        print(f" - {name:5s}: {len(dataset)}개 (기술 스택 없음 비율: {ratio:.1f}%)")
        
    save_jsonl(OUTPUT_DIR / "train.jsonl", train_data)
    save_jsonl(OUTPUT_DIR / "val.jsonl", val_data)
    save_jsonl(OUTPUT_DIR / "test.jsonl", test_data)
    
    print("=" * 50)
    print(f"모든 분할 작업이 완료되었습니다! 파일 저장 위치: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
