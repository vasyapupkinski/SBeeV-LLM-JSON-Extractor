import os
from pathlib import Path
from collections import Counter
import re

def analyze_1000_files():
    raw_dir = Path("data/raw/wanted")
    files = list(raw_dir.glob("*.md"))
    
    header_counter = Counter()
    footer_patterns = [
        "## 이 포지션을 찾고 계셨나요?",
        "본 채용 정보는",
        "지원하기",
        "합격보상금",
        "응답률",
        "팔로우"
    ]
    pattern_positions = {p: [] for p in footer_patterns}
    
    print(f"🧐 {len(files)}개 파일 분석 시작...")
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            # 1. 헤더 분석
            for line in lines:
                if line.startswith("#"):
                    header_counter[line.strip()] += 1
            
            # 2. 특정 패턴 출현 위치 분석
            full_text = "".join(lines)
            for p in footer_patterns:
                idx = full_text.find(p)
                if idx != -1:
                    # 전체 텍스트 중 어느 정도 지점(0~1)에서 나타나는지 기록
                    pattern_positions[p].append(idx / len(full_text))

    print("\n📊 [1] 가장 많이 등장하는 헤더 (Top 10)")
    for header, count in header_counter.most_common(10):
        print(f"   - {header}: {count}개")

    print("\n📉 [2] 노이즈 패턴 출현 빈도 및 평균 위치")
    for p, pos_list in pattern_positions.items():
        if pos_list:
            avg_pos = sum(pos_list) / len(pos_list)
            print(f"   - '{p}': {len(pos_list)}회 발견 (평균 {avg_pos*100:.1f}% 지점)")
        else:
            print(f"   - '{p}': 발견되지 않음")

    # 3. 데이터 손실 없이 잘라낼 수 있는 '안전한 절단 지점' 제안
    print("\n💡 [분석 결론: 전처리 전략]")
    print("1. '## 이 포지션을 찾고 계셨나요?' 이후는 100% 무관한 데이터이므로 무조건 절단.")
    print("2. '본 채용 정보는' 구문도 공고 끝부분의 법적 고지이므로 절단 가능.")
    print("3. 이미지/URL 제거는 기본적으로 수행 (루프 방지)")

if __name__ == "__main__":
    analyze_1000_files()
