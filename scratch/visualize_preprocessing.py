import os
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib

# 서버 환경 및 한글 설정
matplotlib.use('Agg')
font_path = '/mnt/c/Windows/Fonts/malgun.ttf'
fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False

def preprocess_text(text):
    original_len = len(text)
    
    # 1. 노이즈 마커 절단
    noise_markers = ["## 이 포지션을 찾고 계셨나요?", "본 채용 정보는", "지원하기", "## 태그"]
    cut_index = len(text)
    for marker in noise_markers:
        idx = text.find(marker)
        if idx != -1:
            cut_index = min(cut_index, idx)
    text = text[:cut_index]
    
    # 2. 이미지 및 URL 제거
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    
    clean_len = len(text.strip())
    return original_len, clean_len

def visualize_preprocessing_stats():
    raw_dir = Path("data/raw/wanted")
    files = list(raw_dir.glob("*.md"))
    output_dir = Path("data/inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_lengths = []
    clean_lengths = []
    
    print(f"📊 {len(files)}개 파일 데이터 수집 중...")
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            orig, clean = preprocess_text(content)
            original_lengths.append(orig)
            clean_lengths.append(clean)

    # 차트 스타일 설정
    plt.style.use('dark_background')
    
    # --- Chart 1: Length Distribution Comparison ---
    plt.figure(figsize=(12, 7))
    plt.hist(original_lengths, bins=50, alpha=0.5, label='원본 글자수', color='#e74c3c')
    plt.hist(clean_lengths, bins=50, alpha=0.7, label='전처리 후 글자수', color='#2ecc71')
    plt.title('전처리 전/후 공고문 글자수 분포 비교 (1,000건)', fontsize=16, pad=20, color='#f1c40f')
    plt.xlabel('글자수 (Characters)')
    plt.ylabel('공고 개수')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "content_length_comparison.png", dpi=300)
    plt.close()

    # --- Chart 2: Average Reduction Ratio ---
    avg_orig = sum(original_lengths) / len(original_lengths)
    avg_clean = sum(clean_lengths) / len(clean_lengths)
    reduction = ((avg_orig - avg_clean) / avg_orig) * 100

    plt.figure(figsize=(8, 8))
    labels = ['유효 데이터', '제거된 노이즈']
    sizes = [avg_clean, avg_orig - avg_clean]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140, colors=colors)
    plt.title(f'평균 데이터 감소율: {reduction:.1f}%', fontsize=18, pad=20, color='#f1c40f')
    plt.tight_layout()
    plt.savefig(output_dir / "data_reduction_ratio.png", dpi=300)
    plt.close()

    print(f"✅ 시각화 완료! 결과 파일:")
    print(f"   - {output_dir / 'content_length_comparison.png'}")
    print(f"   - {output_dir / 'data_reduction_ratio.png'}")

if __name__ == "__main__":
    visualize_preprocessing_stats()
