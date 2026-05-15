import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "inference_results"

def generate_benchmark_chart():
    models = ['Qwen 3.5\n(Base)', 'GPT-4o-mini\n(General)', 'SBV-LLM-V2\n(Ours)']
    f1_scores = [0.0000, 0.4242, 0.6017]
    latencies = [2.53, 2.36, 2.11]

    # 1. F1 Score Comparison (Higher is better)
    plt.figure(figsize=(10, 6))
    colors = ['#cccccc', '#ff9999', '#66b3ff']
    bars = plt.bar(models, f1_scores, color=colors, width=0.6)
    
    plt.title('Extraction Performance Comparison (F1 Score)', fontsize=15, pad=20)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 0.8) # 점수 분포에 맞춰 조절
    
    # 막대 위에 점수 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_performance_f1.png", dpi=150)
    plt.close()

    # 2. Latency Comparison (Lower is better)
    plt.figure(figsize=(10, 6))
    colors = ['#cccccc', '#ff9999', '#66ff99'] # 속도는 초록색으로 강조
    bars = plt.bar(models, latencies, color=colors, width=0.6)
    
    plt.title('Inference Speed Comparison (Latency)', fontsize=15, pad=20)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.ylim(0, 3.5)
    
    # 막대 위에 수치 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_performance_latency.png", dpi=150)
    plt.close()

    print(f"✅ 최종 벤치마크 그래프 생성 완료!")
    print(f"- 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_benchmark_chart()
