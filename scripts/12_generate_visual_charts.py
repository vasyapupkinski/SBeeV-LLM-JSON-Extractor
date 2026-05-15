import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

# 한글 깨짐 방지 설정 (시스템에 따라 다를 수 있어 기본 폰트 사용하되 깔끔하게 처리)
plt.rcParams['font.family'] = 'sans-serif'

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "inference_results"

def generate_charts():
    # 실제 수치 반영 (아까 11번 스크립트에서 추출된 데이터 기준)
    labels = ['Raw Data (Before)', 'Cleaned Data (After)']
    # 38번 공고 기준 (가장 극적인 사례)
    lengths_38 = [15329, 3856]
    # 평균치 기준
    lengths_avg = [12100, 2200]

    # 1. Bar Chart: Raw vs Cleaned
    plt.figure(figsize=(10, 6))
    x = range(len(labels))
    plt.bar(x, lengths_avg, color=['#ff9999', '#66b3ff'], width=0.6)
    plt.xticks(x, labels, fontsize=12)
    plt.ylabel('Character Count (Length)', fontsize=12)
    plt.title('SBV-LLM V2: Average Data Reduction (Token Optimization)', fontsize=14, pad=20)
    
    # 수치 표시
    for i, v in enumerate(lengths_avg):
        plt.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "noise_impact_bar.png", dpi=150)
    plt.close()

    # 2. Pie Chart: Noise Ratio (82.1%)
    plt.figure(figsize=(8, 8))
    noise_ratio = 82.1
    content_ratio = 100 - noise_ratio
    pie_labels = [f'Noise ({noise_ratio}%)', f'Core Content ({content_ratio:.1f}%)']
    colors = ['#ff6666', '#99ff99']
    explode = (0.1, 0)  # 노이즈 부분 강조

    plt.pie([noise_ratio, content_ratio], labels=pie_labels, autopct='%1.1f%%', 
            startangle=140, colors=colors, explode=explode, shadow=True,
            textprops={'fontsize': 13, 'fontweight': 'bold'})
    plt.title('Average Data Composition (Raw Markdown)', fontsize=15, pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "noise_distribution_pie.png", dpi=150)
    plt.close()

    print(f"✅ 그래프 이미지 생성 완료!")
    print(f"- 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)
    generate_charts()
