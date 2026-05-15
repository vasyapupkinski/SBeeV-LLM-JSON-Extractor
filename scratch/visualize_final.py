import json
import matplotlib.pyplot as plt
from pathlib import Path

# 폰트 설정을 기본값으로 되돌려 Linux/WSL 호환성 확보
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "final_benchmark.json"

def main():
    if not RESULTS_FILE.exists():
        print("결과 파일이 없습니다.")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = list(data.keys())
    f1_scores = [d["f1"] for d in data.values()]
    latencies = [d["latency"] for d in data.values()]

    # 그래프 생성
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # F1 Score 막대 그래프
    bars = ax1.bar(models, f1_scores, color=['#A0A0A0', '#4A90E2', '#FF6B6B'], alpha=0.7, label='F1 Score')
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.set_title('SBV-LLM Performance Benchmark (F1-Score & Latency)', fontsize=14)

    # 점수 텍스트 표시
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Latency 꺾은선 그래프 (이중 축)
    ax2 = ax1.twinx()
    ax2.plot(models, latencies, color='green', marker='o', linestyle='--', linewidth=2, label='Latency (s)')
    ax2.set_ylabel('Latency (seconds)', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')

    plt.tight_layout()
    
    # 이미지 저장
    save_path = PROJECT_ROOT / "results" / "performance_chart.png"
    plt.savefig(save_path, dpi=300)
    print(f"차트가 재생성되었습니다: {save_path}")

if __name__ == "__main__":
    main()
