import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wanted"
# 전처리 로직 (스크립트에 있는 것과 동일하게 복사)
def clean_text_with_stats(text):
    original_len = len(text)
    
    # 1. 이미지 개수 파악
    images = re.findall(r'!\[.*?\]\(.*?\)', text)
    img_count = len(images)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 2. 링크 개수 파악
    links = re.findall(r'\[(.*?)\]\(.*?\)', text)
    link_count = len(links)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # 3. 푸터 노이즈 마커 이후 제거
    noise_patterns = [r"## 이 포지션을 찾고 계셨나요\?", r"본\s*채용\s*정보는", r"지원하기", r"서류 합격 확률이 아주 높아요"]
    cut_index = original_len
    for pattern in noise_patterns:
        match = re.search(pattern, text)
        if match:
            cut_index = min(cut_index, match.start())
    
    footer_removed_len = len(text) - cut_index if cut_index < len(text) else 0
    text = text[:cut_index].strip()
    
    final_len = len(text)
    noise_removed = original_len - final_len
    
    return {
        "original_len": original_len,
        "final_len": final_len,
        "noise_removed": noise_removed,
        "img_count": img_count,
        "link_count": link_count,
        "footer_removed_len": footer_removed_len,
        "reduction_pct": (noise_removed / original_len) * 100 if original_len > 0 else 0
    }

def main():
    # 문제가 되었던 샘플들을 포함해 상위 10개 분석
    sample_files = sorted(list(RAW_DIR.glob("*.md")))[:10]
    
    # 특히 문제가 되었던 38번(354263.md) 추가 분석
    target_38 = RAW_DIR / "354263.md"
    if target_38.exists() and target_38 not in sample_files:
        sample_files.append(target_38)

    print(f"{'File Name':<15} | {'Original':>8} | {'Cleaned':>8} | {'Noise %':>8} | {'Img/Link':>8}")
    print("-" * 65)

    total_reduction = 0
    for file_path in sample_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        stats = clean_text_with_stats(content)
        total_reduction += stats['reduction_pct']
        
        print(f"{file_path.name:<15} | {stats['original_len']:>8} | {stats['final_len']:>8} | {stats['reduction_pct']:>7.1f}% | {stats['img_count']}/{stats['link_count']}")

    avg_reduction = total_reduction / len(sample_files)
    print("-" * 65)
    print(f"평균 노이즈 제거율: {avg_reduction:.1f}%")
    print("\n[포트폴리오용 인사이트]")
    print(f"가장 문제가 되었던 38번 파일의 경우, 전체 데이터의 {clean_text_with_stats(open(target_38, 'r', encoding='utf-8').read())['reduction_pct']:.1f}%가 모델 추론에 방해되는 노이즈였습니다.")

if __name__ == "__main__":
    main()
