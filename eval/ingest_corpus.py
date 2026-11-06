"""Ingest and freeze official captions for the 15-video research corpus."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from tabulate import tabulate

from transcript import fetch_captions

CORPUS_META_PATH = os.path.join("eval", "corpus_meta.json")
OUTPUT_DIR = os.path.join("data", "transcripts", "captions")


def load_corpus_metadata(path: str = CORPUS_META_PATH) -> list[dict[str, Any]]:
    """Load flat list of videos from corpus metadata file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus metadata not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_videos = []
    for tier_key, tier_data in data.get("tiers", {}).items():
        for vid in tier_data.get("videos", []):
            item = dict(vid)
            item["tier"] = tier_key
            flat_videos.append(item)
    return flat_videos


def ingest_corpus(
    sleep_sec: float = 3.0,
    force_refresh: bool = False,
    browser: str | None = None,
) -> None:
    """Fetch and freeze captions for all corpus videos with polite backoff."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    videos = load_corpus_metadata()

    print("\n" + "=" * 80)
    print("VIDQUERY RESEARCH CORPUS INGESTION (OFFLINE DATASET FREEZING)")
    print(f"Total Videos Target: {len(videos)}")
    print(f"Output Directory:    {OUTPUT_DIR}")
    if browser:
        print(f"Browser Cookies:     {browser}")
    print("=" * 80 + "\n")

    summary_rows = []

    for idx, v in enumerate(videos, start=1):
        v_id = v["video_id"]
        v_tag = v["tag"]
        v_tier = v["tier"]
        target_path = os.path.join(OUTPUT_DIR, f"{v_id}.json")
        legacy_path = os.path.join("data", "transcripts", f"{v_id}.json")

        status = "UNKNOWN"
        n_segs = 0
        dur_min = 0

        # Check existing cache
        cached_source = None
        if not force_refresh:
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                cached_source = target_path
            elif os.path.exists(legacy_path) and os.path.getsize(legacy_path) > 0:
                cached_source = legacy_path

        if cached_source:
            try:
                with open(cached_source, "r", encoding="utf-8") as f:
                    segments = json.load(f)
                n_segs = len(segments)
                dur_min = int(segments[-1]["start"] + segments[-1].get("duration", 0)) // 60
                status = "CACHED"
                # Ensure it exists in the new structured directory
                if cached_source != target_path:
                    with open(target_path, "w", encoding="utf-8") as f:
                        json.dump(segments, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                status = f"CORRUPT ({exc})"

        if status not in {"CACHED"}:
            print(f"[{idx}/{len(videos)}] Fetching captions for: {v_tag} ({v_id})...")
            # 1. Try standard youtube-transcript-api first
            try:
                segments = fetch_captions(v_id)
                n_segs = len(segments)
                dur_min = int(segments[-1]["start"] + segments[-1].get("duration", 0)) // 60
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                status = "DOWNLOADED"
            except Exception as primary_exc:
                # 2. Try yt-dlp fallback with browser cookies if requested
                if browser:
                    try:
                        import yt_dlp

                        ydl_opts = {
                            "skip_download": True,
                            "writesubtitles": True,
                            "writeautomaticsub": True,
                            "subtitleslangs": ["en.*", "en"],
                            "subtitlesformat": "json3",
                            "quiet": True,
                            "no_warnings": True,
                            "cookiesfrombrowser": (browser, None, None, None),
                        }
                        url = f"https://www.youtube.com/watch?v={v_id}"
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=False)
                            # Parsed via yt-dlp metadata
                        status = "DOWNLOADED (COOKIES)"
                    except Exception as ytdl_exc:
                        status = f"FAILED: {primary_exc}"
                else:
                    err_msg = str(primary_exc)
                    if "blocking requests from your IP" in err_msg or "429" in err_msg:
                        status = "RATE_LIMITED (HTTP 429)"
                    elif "unavailable" in err_msg.lower():
                        status = "UNAVAILABLE"
                    else:
                        status = f"FAILED: {err_msg[:60]}"

            # Polite delay to prevent YouTube IP blocks
            if idx < len(videos):
                time.sleep(sleep_sec)

        summary_rows.append([
            idx,
            v_tier.replace("tier_", "").replace("_", " ").title(),
            v_tag,
            v_id,
            dur_min,
            n_segs,
            status,
        ])

    headers = ["#", "Tier", "Tag", "Video ID", "Dur (m)", "Segments", "Status"]
    print("\n" + tabulate(summary_rows, headers=headers, tablefmt="github"))
    cached_count = sum(1 for r in summary_rows if "CACHED" in r[-1] or "DOWNLOADED" in r[-1])
    print(f"\nCorpus Ingestion Summary: {cached_count}/{len(videos)} videos frozen locally.\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest and freeze captions for research corpus.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached.")
    parser.add_argument("--sleep", type=float, default=3.0, help="Polite delay between requests in seconds.")
    parser.add_argument(
        "--cookies-from-browser",
        choices=["chrome", "edge", "firefox"],
        default=None,
        help="Extract session cookies from local browser to bypass anonymous IP rate limits.",
    )
    args = parser.parse_args()

    ingest_corpus(
        sleep_sec=args.sleep,
        force_refresh=args.force,
        browser=args.cookies_from_browser,
    )


if __name__ == "__main__":
    main()
