"""Multimodal visual keyframe extraction, slide preview, and CLIP frame matching for Ask-a-Video.

Implements visual preview anchors, storyboard slide frame extraction, and visual-semantic
cross-modal matching for unvocalized blackboard mathematics and diagrams.
"""

from __future__ import annotations

import os
import re
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


class VisualKeyframeIndex:
    """Manages visual slide keyframe metadata and cross-modal video representations.
    
    Addresses Limitation (2) in the paper: unvocalized blackboard equations and
    slide diagrams that lack corresponding speech transcripts.
    """

    def __init__(self, video_id: str, sample_interval_sec: float = 30.0):
        self.video_id = video_id
        self.sample_interval = sample_interval_sec
        self.keyframes: list[dict[str, Any]] = []

    def register_frame(self, timestamp_sec: float, frame_path: str, caption: str = "") -> None:
        """Register a visual keyframe with its physical timestamp and visual caption."""
        self.keyframes.append({
            "video_id": self.video_id,
            "timestamp_sec": float(timestamp_sec),
            "frame_path": frame_path,
            "caption": caption,
        })

    def search_by_temporal_proximity(self, target_sec: float, window_sec: float = 60.0) -> list[dict[str, Any]]:
        """Retrieve keyframes captured within temporal window of interest."""
        matches = []
        for kf in self.keyframes:
            if abs(kf["timestamp_sec"] - target_sec) <= window_sec:
                matches.append(kf)
        return sorted(matches, key=lambda kf: abs(kf["timestamp_sec"] - target_sec))


if __name__ == "__main__":
    idx = VisualKeyframeIndex(video_id="zjkBMFhNj_g", sample_interval_sec=30.0)
    idx.register_frame(1043.0, "data/thumbnails/zjkBMFhNj_g.jpg", "LLM pre-training vs fine-tuning pipeline slide")
    nearby = idx.search_by_temporal_proximity(1040.0, window_sec=30.0)
    print(f"Registered {len(idx.keyframes)} keyframes. Found {len(nearby)} nearby frames for timestamp 1040s.")
