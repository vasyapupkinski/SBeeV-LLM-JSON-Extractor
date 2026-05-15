"""
01_crawl_wanted.py — 원티드 채용 공고 크롤러 (API 기반)

원티드 공개 API를 직접 호출하여 공고 ID를 수집하고,
각 상세 페이지를 마크다운으로 저장한다.

API 기반이므로 무한 스크롤 문제 없음, 브라우저 렌더링 불필요.
상세 페이지만 Crawl4AI로 스크래핑.

Usage:
    python scripts/01_crawl_wanted.py
"""

import asyncio
import json
import time
from pathlib import Path

import httpx
import yaml
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

# ── 설정 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "crawl_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "wanted"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "raw" / "wanted_checkpoint.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CRAWLER_CFG = config["crawler"]
TARGET_COUNT = config["sources"]["wanted"]["target_count"]
DELAY = CRAWLER_CFG["delay_seconds"]

# 원티드 공개 API
WANTED_API = "https://www.wanted.co.kr/api/v4/jobs"
API_PARAMS = {
    "country": "kr",
    "tag_type_ids": "518",  # IT/개발
    "job_sort": "job.latest_order",
    "years": "-1",
    "locations": "all",
}
PAGE_SIZE = 20


# ── 체크포인트 ────────────────────────────────────────
def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"collected_ids": [], "scraped_ids": [], "failed_ids": []}


def save_checkpoint(cp: dict):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


# ── Step 1: API로 공고 ID 수집 ────────────────────────
async def collect_job_ids(cp: dict) -> list[int]:
    """원티드 API를 페이지네이션하여 공고 ID를 수집."""
    if len(cp["collected_ids"]) >= TARGET_COUNT:
        print(f"[체크포인트] 이전 수집 {len(cp['collected_ids'])}건 재사용")
        return cp["collected_ids"]

    all_ids = cp["collected_ids"]  # 기존 수집분 유지
    offset = len(all_ids)          # 기존 개수부터 시작
    
    if all_ids:
        print(f"[체크포인트] 기존 {len(all_ids)}건에 이어서 추가 수집 시작 (목표: {TARGET_COUNT})")

    async with httpx.AsyncClient(
        headers={"User-Agent": CRAWLER_CFG["user_agent"]},
        timeout=30,
    ) as client:
        while len(all_ids) < TARGET_COUNT:
            params = {**API_PARAMS, "limit": PAGE_SIZE, "offset": offset}
            print(f"[API] offset={offset}, 누적={len(all_ids)}건")

            try:
                resp = await client.get(WANTED_API, params=params)
                resp.raise_for_status()
                data = resp.json()

                jobs = data.get("data", [])
                if not jobs:
                    print("  → 더 이상 공고 없음")
                    break

                for job in jobs:
                    all_ids.append(job["id"])

                offset += PAGE_SIZE

                # 다음 페이지가 없으면 종료
                if not data.get("links", {}).get("next"):
                    break

            except Exception as e:
                print(f"  [오류] {e}")
                break

            await asyncio.sleep(0.5)  # API는 딜레이 짧게

    all_ids = all_ids[:TARGET_COUNT]
    cp["collected_ids"] = all_ids
    save_checkpoint(cp)
    print(f"[완료] 공고 ID {len(all_ids)}건 수집")
    return all_ids


# ── Step 2: 상세 페이지 스크래핑 ──────────────────────
async def scrape_detail_pages(crawler: AsyncWebCrawler, job_ids: list[int], cp: dict):
    """각 공고 상세 페이지를 마크다운으로 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scraped = set(str(i) for i in cp["scraped_ids"])
    failed = set(str(i) for i in cp["failed_ids"])

    run_config = CrawlerRunConfig(
        delay_before_return_html=2.0,
    )

    total = len(job_ids)
    for i, job_id in enumerate(job_ids):
        sid = str(job_id)
        if sid in scraped:
            continue

        url = f"https://www.wanted.co.kr/wd/{job_id}"
        print(f"[{i+1}/{total}] 스크래핑: {url}")

        try:
            result = await crawler.arun(url=url, config=run_config)

            if result.success and result.markdown and len(result.markdown) > 200:
                filepath = OUTPUT_DIR / f"{job_id}.md"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"<!-- source: {url} -->\n")
                    f.write(f"<!-- scraped_at: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n\n")
                    f.write(result.markdown)

                scraped.add(sid)
                print(f"  ✅ 저장 ({len(scraped)}건)")
            else:
                failed.add(sid)
                print(f"  ⚠️ 콘텐츠 부족")

        except Exception as e:
            failed.add(sid)
            print(f"  ❌ 오류: {e}")

        # 체크포인트 (20건마다)
        if len(scraped) % 20 == 0:
            cp["scraped_ids"] = list(scraped)
            cp["failed_ids"] = list(failed)
            save_checkpoint(cp)

        await asyncio.sleep(DELAY)

    cp["scraped_ids"] = list(scraped)
    cp["failed_ids"] = list(failed)
    save_checkpoint(cp)

    print(f"\n{'='*50}")
    print(f"[결과] 성공: {len(scraped)}건 / 실패: {len(failed)}건")


# ── Main ──────────────────────────────────────────────
async def main():
    print("=" * 50)
    print("SBV-LLM 원티드 크롤러 (API 기반)")
    print(f"목표: {TARGET_COUNT}건")
    print("=" * 50)

    cp = load_checkpoint()

    # Step 1: API로 ID 수집 (브라우저 불필요)
    job_ids = await collect_job_ids(cp)

    # Step 2: 상세 페이지만 Crawl4AI로 스크래핑
    browser_config = BrowserConfig(
        headless=True,
        user_agent=CRAWLER_CFG["user_agent"],
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        await scrape_detail_pages(crawler, job_ids, cp)


if __name__ == "__main__":
    asyncio.run(main())
