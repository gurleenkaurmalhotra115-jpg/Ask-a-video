"""Transcript fetching and fallback transcription module for VidQuery."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import parse_qs, urlparse


class TranscriptError(Exception):
    """Custom exception raised with a user-friendly message when transcript operations fail."""
    pass


def fmt_time(seconds: float | int) -> str:
    """Format seconds into 'mm:ss' or 'h:mm:ss' display format."""
    total_seconds = int(max(0, round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_video_id(url: str) -> str:
    """Extract 11-character YouTube video ID from supported URL patterns or bare ID.
    
    Supports:
      - watch?v=ID
      - youtu.be/ID
      - /shorts/ID
      - /embed/ID
      - bare 11-character ID
    
    Raises:
      TranscriptError: If input does not contain a valid YouTube video ID.
    """
    if not url or not isinstance(url, str):
        raise TranscriptError("Please provide a valid YouTube URL or video ID.")

    cleaned = url.strip()

    # 1. Check if input is already a bare 11-character video ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", cleaned):
        return cleaned

    # 2. Parse URL components
    try:
        parsed = urlparse(cleaned)
    except Exception as exc:
        raise TranscriptError(f"Invalid URL structure: {exc}") from exc

    # 3. Check standard watch URLs (youtube.com/watch?v=...)
    if "youtube.com" in parsed.netloc and parsed.path == "/watch":
        query_params = parse_qs(parsed.query)
        v_param = query_params.get("v")
        if v_param and re.fullmatch(r"[a-zA-Z0-9_-]{11}", v_param[0]):
            return v_param[0]

    # 4. Check youtu.be short URLs
    if "youtu.be" in parsed.netloc:
        path_id = parsed.path.lstrip("/").split("/")[0].split("?")[0]
        if re.fullmatch(r"[a-zA-Z0-9_-]{11}", path_id):
            return path_id

    # 5. Check /shorts/ID or /embed/ID
    match = re.search(r"/(?:shorts|embed)/([a-zA-Z0-9_-]{11})", parsed.path)
    if match:
        return match.group(1)

    # 6. Fallback regex across the entire string (e.g. mobile or query embedded)
    regex_patterns = [
        r"[?&]v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"/(?:shorts|embed)/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in regex_patterns:
        m = re.search(pattern, cleaned)
        if m:
            return m.group(1)

    raise TranscriptError(
        f"Could not extract a valid YouTube video ID from: '{url}'. "
        "Please provide a standard YouTube link or an 11-character video ID."
    )


def get_cached_transcript_path(video_id: str, source: str = "captions") -> str | None:
    """Find local cached transcript file checking absolute and relative project paths."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "data", "transcripts", source, f"{video_id}.json"),
        os.path.join(os.getcwd(), "data", "transcripts", source, f"{video_id}.json"),
        os.path.join("data", "transcripts", source, f"{video_id}.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def fetch_captions(video_id: str) -> list[dict[str, Any]]:
    """Fetch captions via local cache or fallback to youtube-transcript-api.
    
    Prefers English and Hindi variants (["en", "en-IN", "en-US", "en-GB", "hi"]),
    falling back to the first available transcript if none match.
    
    Returns:
      List of dicts: {"start": float, "duration": float, "text": str}
      
    Raises:
      TranscriptError: On disabled transcripts, unavailable video, network issues, or no captions.
    """
    # 0. Check local disk cache first (offline-first execution)
    cached_path = get_cached_transcript_path(video_id, "captions")
    if cached_path:
        try:
            with open(cached_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if cached_data and isinstance(cached_data, list):
                return cached_data
        except Exception:
            pass

    try:
        from youtube_transcript_api import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
            YouTubeTranscriptApi,
        )
    except ImportError as exc:
        raise TranscriptError(
            "youtube-transcript-api is not installed. Please install project dependencies."
        ) from exc

    preferred_languages = ["en", "en-IN", "en-US", "en-GB", "hi"]
    raw_data: Any = None

    try:
        api = YouTubeTranscriptApi()
    except TypeError:
        api = YouTubeTranscriptApi

    fetch_method = getattr(api, "fetch", None) or getattr(YouTubeTranscriptApi, "get_transcript", None)
    list_method = getattr(api, "list", None) or getattr(YouTubeTranscriptApi, "list_transcripts", None)

    if fetch_method is None:
        raise TranscriptError("youtube-transcript-api is missing fetch() or get_transcript().")

    # Try preferred languages first
    try:
        raw_data = fetch_method(video_id, languages=preferred_languages)
    except Exception as primary_exc:
        exc_name = primary_exc.__class__.__name__
        exc_msg = str(primary_exc)

        # Fatal errors that cannot be solved by trying another language
        if (TranscriptsDisabled and isinstance(primary_exc, TranscriptsDisabled)) or "TranscriptsDisabled" in exc_name:
            raise TranscriptError("Transcripts are disabled for this video.") from primary_exc
        if (VideoUnavailable and isinstance(primary_exc, VideoUnavailable)) or "VideoUnavailable" in exc_name:
            raise TranscriptError("This video is unavailable (private, deleted, or region-restricted).") from primary_exc
        if any(term in exc_msg.lower() for term in ["connection", "timed out", "timeout", "network", "max retries"]):
            raise TranscriptError("Network error while connecting to YouTube transcript service.") from primary_exc

        # If preferred language not found, try fallback to first available transcript
        if list_method is None:
            raise TranscriptError(f"Failed to fetch captions: {exc_msg}") from primary_exc

        try:
            transcript_list = list_method(video_id)
            first_transcript = None
            for item in transcript_list:
                first_transcript = item
                break

            if first_transcript is None:
                raise TranscriptError("No captions found for this video.")

            raw_data = first_transcript.fetch()
        except Exception as list_exc:
            l_name = list_exc.__class__.__name__
            l_msg = str(list_exc)
            if (TranscriptsDisabled and isinstance(list_exc, TranscriptsDisabled)) or "TranscriptsDisabled" in l_name:
                raise TranscriptError("Transcripts are disabled for this video.") from list_exc
            if (VideoUnavailable and isinstance(list_exc, VideoUnavailable)) or "VideoUnavailable" in l_name:
                raise TranscriptError("This video is unavailable (private, deleted, or region-restricted).") from list_exc
            if (NoTranscriptFound and isinstance(list_exc, NoTranscriptFound)) or "NoTranscriptFound" in l_name:
                raise TranscriptError("No captions found for this video in any language.") from list_exc
            if any(term in l_msg.lower() for term in ["connection", "timed out", "timeout", "network"]):
                raise TranscriptError("Network error while fetching available transcript list.") from list_exc
            raise TranscriptError(f"Could not retrieve captions: {l_msg}") from list_exc

    # Process and normalize segments
    items = raw_data.snippets if hasattr(raw_data, "snippets") else raw_data
    segments: list[dict[str, Any]] = []

    for item in items:
        if isinstance(item, dict):
            start = float(item.get("start", item.get("start_time", 0.0)))
            duration = float(item.get("duration", 0.0))
            raw_text = str(item.get("text", ""))
        else:
            start = float(getattr(item, "start", getattr(item, "start_time", 0.0)))
            duration = float(getattr(item, "duration", 0.0))
            raw_text = str(getattr(item, "text", ""))

        collapsed_text = " ".join(raw_text.split())
        if collapsed_text:
            segments.append({
                "start": round(start, 2),
                "duration": round(duration, 2),
                "text": collapsed_text,
            })

    if not segments:
        raise TranscriptError("Transcript was found but contained no readable text segments.")

    try:
        os.makedirs(os.path.dirname(local_captions_path), exist_ok=True)
        with open(local_captions_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return segments


def transcribe_with_whisper(video_id: str, model_size: str = "base") -> list[dict[str, Any]]:
    """Download audio using yt-dlp and transcribe using local OpenAI Whisper.
    
    Requires ffmpeg to be installed and available in PATH.
    
    Returns:
      List of dicts: {"start": float, "duration": float, "text": str}
      
    Raises:
      TranscriptError: On missing libraries, missing ffmpeg, download failure, or transcription error.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise TranscriptError(
            "yt-dlp is not installed. Please install dependencies from requirements.txt to use Whisper."
        ) from exc

    try:
        import whisper
    except ImportError as exc:
        raise TranscriptError(
            "openai-whisper is not installed. Please install dependencies from requirements.txt to use Whisper."
        ) from exc

    if shutil.which("ffmpeg") is None:
        raise TranscriptError(
            "ffmpeg was not found in your system PATH. Whisper audio extraction requires ffmpeg."
        )

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_template = os.path.join(tmpdir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_template,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
        except Exception as exc:
            raise TranscriptError(f"Failed to download audio for video {video_id}: {exc}") from exc

        downloaded_files = [
            os.path.join(tmpdir, f)
            for f in os.listdir(tmpdir)
            if os.path.isfile(os.path.join(tmpdir, f))
        ]
        if not downloaded_files:
            raise TranscriptError("Audio download completed but no audio file was found.")

        audio_path = downloaded_files[0]

        try:
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_path)
        except FileNotFoundError as exc:
            raise TranscriptError(
                "ffmpeg executable not found during Whisper transcription. Please ensure ffmpeg is in your PATH."
            ) from exc
        except Exception as exc:
            msg = str(exc)
            if "ffmpeg" in msg.lower():
                raise TranscriptError(
                    "ffmpeg is required for Whisper transcription. Please ensure ffmpeg is installed and on your PATH."
                ) from exc
            raise TranscriptError(f"Whisper transcription failed: {msg}") from exc

    segments: list[dict[str, Any]] = []
    for item in result.get("segments", []):
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        duration = max(0.0, end - start)
        collapsed_text = " ".join(str(item.get("text", "")).split())
        if collapsed_text:
            segments.append({
                "start": round(start, 2),
                "duration": round(duration, 2),
                "text": collapsed_text,
            })

    if not segments:
        raise TranscriptError("Whisper completed transcription but no text segments were detected.")

    return segments


def get_transcript(
    url: str, allow_whisper: bool = True
) -> tuple[str, list[dict[str, Any]], str]:
    """Retrieve video transcript, attempting local disk cache first, then official captions, with optional Whisper fallback.
    
    Args:
      url: YouTube URL or 11-char video ID.
      allow_whisper: If True, falls back to Whisper when captions are not available.
      
    Returns:
      tuple: (video_id, segments, source) where source is 'captions' or 'whisper'.
      
    Raises:
      TranscriptError: If transcript extraction fails.
    """
    video_id = extract_video_id(url)

    # 1. Check local frozen captions cache first
    cached_captions = get_cached_transcript_path(video_id, "captions")
    if cached_captions:
        try:
            with open(cached_captions, "r", encoding="utf-8") as f:
                segments = json.load(f)
            if segments and isinstance(segments, list):
                return video_id, segments, "captions"
        except Exception:
            pass

    # 2. Check local frozen whisper cache if present
    cached_whisper = get_cached_transcript_path(video_id, "whisper")
    if cached_whisper:
        try:
            with open(cached_whisper, "r", encoding="utf-8") as f:
                segments = json.load(f)
            if segments and isinstance(segments, list):
                return video_id, segments, "whisper"
        except Exception:
            pass

    caption_error: Exception | None = None

    try:
        segments = fetch_captions(video_id)
        return video_id, segments, "captions"
    except TranscriptError as exc:
        caption_error = exc

    if allow_whisper:
        try:
            segments = transcribe_with_whisper(video_id)
            target_whisper_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data",
                "transcripts",
                "whisper",
                f"{video_id}.json",
            )
            try:
                os.makedirs(os.path.dirname(target_whisper_path), exist_ok=True)
                with open(target_whisper_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
            return video_id, segments, "whisper"
        except TranscriptError as whisper_err:
            raise TranscriptError(
                f"Captions unavailable ({caption_error}). Whisper fallback also failed: {whisper_err}"
            ) from whisper_err

    if caption_error is not None:
        raise caption_error
    raise TranscriptError("Unable to retrieve transcript for video.")
