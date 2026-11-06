"""Multimodal visual keyframe extraction and thumbnail helper for VidQuery.

Implements visual preview anchors and slide frame extraction for timestamped video chunks.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.request import urlopen, Request


def get_youtube_thumbnail_urls(video_id: str) -> dict[str, str]:
    """Return dictionary of available YouTube thumbnail image resolutions."""
    return {
        "maxres": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        "hq": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "mq": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        "storyboard_0": f"https://img.youtube.com/vi/{video_id}/0.jpg",
        "storyboard_1": f"https://img.youtube.com/vi/{video_id}/1.jpg",
        "storyboard_2": f"https://img.youtube.com/vi/{video_id}/2.jpg",
        "storyboard_3": f"https://img.youtube.com/vi/{video_id}/3.jpg",
    }


def get_timestamp_embed_url(video_id: str, timestamp_sec: float | int, autoplay: bool = False) -> str:
    """Generate embeddable YouTube player URL with exact start seconds parameter."""
    start_int = int(max(0, round(timestamp_sec)))
    ap_flag = "1" if autoplay else "0"
    return f"https://www.youtube.com/embed/{video_id}?start={start_int}&autoplay={ap_flag}&rel=0"


def download_thumbnail(video_id: str, output_dir: str = "data/thumbnails") -> str | None:
    """Download best available video thumbnail to local disk cache.
    
    Returns:
      Path to saved thumbnail on disk, or None if network fetch fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, f"{video_id}.jpg")

    if os.path.isfile(target_path):
        return target_path

    urls = get_youtube_thumbnail_urls(video_id)
    # Try maxres first, fall back to hq
    for resolution in ["maxres", "hq", "mq"]:
        url = urls[resolution]
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = response.read()
                    # YouTube returns a 120-byte dummy transparent image if maxres is unavailable
                    if len(data) > 1000:
                        with open(target_path, "wb") as f:
                            f.write(data)
                        return target_path
        except Exception:
            continue

    return None
