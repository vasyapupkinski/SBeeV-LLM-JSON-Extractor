import json
import os
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "data" / "inference_results" / "wanted_v2_results.json"
REPORT_FILE = PROJECT_ROOT / "data" / "inference_results" / "v2_test_summary.md"

def main():
    if not RESULTS_FILE.exists():
        print(f"❌ 결과 파일을 찾을 수 없습니다: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)
    success_count = sum(1 for r in results if r["is_valid_json"])
    fail_count = total - success_count
    success_rate = (success_count / total) * 100 if total > 0 else 0

    # 에러 유형 분석
    errors = [r["error"] for r in results if not r["is_valid_json"]]
    error_summary = Counter()
    for err in errors:
        if "Unterminated string" in err:
            error_summary["Looping (Token Limit)"] += 1
        elif "Expecting value" in err:
            error_summary["Empty/Invalid JSON"] += 1
        else:
            error_summary[err[:50]] += 1

    # 리포트 생성
    report = []
    report.append("# SBV-LLM V2 Inference Test Statistics")
    report.append(f"\n- **Total Samples:** {total}")
    report.append(f"- **Success:** {success_count} ✅")
    report.append(f"- **Failure:** {fail_count} ❌")
    report.append(f"- **Success Rate:** {success_rate:.1f}%")

    report.append("\n## 📊 Success Rate Chart")
    report.append("```mermaid")
    report.append("pie title V2 Extraction Success Rate")
    report.append(f'    "Success ({success_count})" : {success_count}')
    report.append(f'    "Failure ({fail_count})" : {fail_count}')
    report.append("```")

    if error_summary:
        report.append("\n## ⚠️ Error Analysis")
        report.append("| Error Type | Count | Percentage |")
        report.append("| :--- | :---: | :---: |")
        for err_type, count in error_summary.items():
            pct = (count / fail_count) * 100 if fail_count > 0 else 0
            report.append(f"| {err_type} | {count} | {pct:.1f}% |")

        report.append("\n## 📈 Error Distribution")
        report.append("```mermaid")
        report.append("graph LR")
        for i, (err_type, count) in enumerate(error_summary.items()):
            report.append(f'    E{i}["{err_type} ({count})"]')
        report.append("```")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n✅ 통계 분석 완료!")
    print(f"- 성공률: {success_rate:.1f}% ({success_count}/{total})")
    print(f"- 리포트 저장 위치: {REPORT_FILE}")

if __name__ == "__main__":
    main()
