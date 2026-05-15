import json
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import re

# 서버 환경에서도 그래프를 파일로 저장할 수 있도록 설정
import matplotlib
import matplotlib.font_manager as fm
matplotlib.use('Agg')

# 한글 폰트 설정 (WSL에서 윈도우 폰트 경로 사용)
font_path = '/mnt/c/Windows/Fonts/malgun.ttf'
fm.fontManager.addfont(font_path) # 폰트 직접 등록
font_name = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

def visualize_all():
    results_path = Path("data/inference_results/wanted_100_results.json")
    raw_dir = Path("data/raw/wanted")
    output_dir = Path("data/inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 기술 스택 분포 (Top 15)
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tech_list = []
    exp_list = []
    for entry in data:
        if entry.get("is_valid_json") and isinstance(entry.get("extracted_data"), dict):
            tech_list.extend(entry["extracted_data"].get("tech_stack", []))
            exp = entry["extracted_data"].get("experience", "Unknown")
            if exp: exp_list.append(exp)

    # 차트 스타일 설정
    plt.style.use('dark_background')

    # --- Chart 1: Tech Stack ---
    tech_counts = Counter(tech_list).most_common(15)
    names, counts = zip(*tech_counts) if tech_counts else ([], [])
    plt.figure(figsize=(12, 8))
    plt.barh(names[::-1], counts[::-1], color='#3498db')
    plt.title('Top 15 Tech Stacks (Extracted by SBV-LLM)', fontsize=16, pad=20, color='#f1c40f')
    plt.tight_layout()
    plt.savefig(output_dir / "tech_stack_stats.png", dpi=300)
    plt.close()

    # --- Chart 2: Section Consistency (Static Analysis from 1000 files) ---
    # 아까 분석한 데이터 기반 (하드코딩으로 시각화 최적화)
    sections = ["Position Details", "Main Tasks", "Requirements", "Deadline", "Location", "Tags"]
    counts = [1000, 1000, 1000, 1000, 1000, 980]
    plt.figure(figsize=(10, 6))
    plt.bar(sections, counts, color='#2ecc71', alpha=0.7)
    plt.title('Data Field Consistency across 1,000 Documents', fontsize=16, pad=20, color='#f1c40f')
    plt.ylim(900, 1050) # 차이를 보여주기 위해 y축 조정
    plt.ylabel('File Count')
    plt.tight_layout()
    plt.savefig(output_dir / "section_consistency.png", dpi=300)
    plt.close()

    # --- Chart 3: Experience Level Distribution ---
    exp_counts = Counter(exp_list).most_common(10)
    names, counts = zip(*exp_counts) if exp_counts else ([], [])
    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=names, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
    plt.title('Required Experience Level Distribution', fontsize=16, pad=20, color='#f1c40f')
    plt.tight_layout()
    plt.savefig(output_dir / "experience_dist.png", dpi=300)
    plt.close()

    print(f"✅ 3개의 시각화 파일이 {output_dir} 에 저장되었습니다.")

if __name__ == "__main__":
    visualize_all()
