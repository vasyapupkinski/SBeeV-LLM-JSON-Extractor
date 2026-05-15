import os
from pathlib import Path
from collections import Counter
import re

def analyze_headers():
    raw_dir = Path("data/raw/wanted")
    if not raw_dir.exists():
        print("Data directory not found.")
        return

    header_counter = Counter()
    
    # Analyze up to 1000 files
    files = sorted(list(raw_dir.glob("*.md")))[:1000]
    
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Find lines starting with ## or ###
                headers = re.findall(r'^(#{2,3}\s.*)$', content, re.MULTILINE)
                for h in headers:
                    header_counter[h.strip()] += 1
        except:
            continue

    print("\n[전체 공고 섹션 분석 결과 (TOP 20)]")
    print("-" * 50)
    for header, count in header_counter.most_common(20):
        print(f"{count:4d}회 등장 | {header}")

if __name__ == "__main__":
    analyze_headers()
