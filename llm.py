"""LLM grounded question answering with retries, fallback model, disk caching, and friendly errors."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import random
import re
import time
from typing import Any, Callable

from dotenv import load_dotenv

from transcript import fmt_time

logger = logging.getLogger(__name__)

CACHE_FILE_DEFAULT = "cache/llm_cache.json"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """User-readable exception for LLM configuration, invocation, or rate limit issues."""
    pass


SYSTEM_PROMPT = (
    "You answer questions about a video using ONLY the numbered transcript chunks provided. "
    "Explain like a clear tutor, concisely. After each claim, cite the chunk number in square brackets, "
    "like [1] or [2][3]. NEVER write timestamps yourself, only chunk numbers. "
    "If the chunks do not contain the answer, reply with exactly: NOT_IN_VIDEO"
)


_client_instance: Any = None


def get_client() -> Any:
    """Load environment variables and initialize the Google GenAI client (cached singleton).

    Raises:
        LLMError: If GEMINI_API_KEY is not defined in the environment or .env file.
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise LLMError("Add GEMINI_API_KEY to your .env file")

    try:
        from google import genai
        _client_instance = genai.Client(api_key=api_key.strip())
        return _client_instance
    except Exception as exc:
        raise LLMError(f"Failed to initialize Gemini client: {exc}") from exc


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into numbered context blocks: [n] (mm:ss-mm:ss) <text>."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        t_start = fmt_time(chunk["start"])
        t_end = fmt_time(chunk["end"])
        blocks.append(f"[{i}] ({t_start}-{t_end}) {chunk['text']}")
    return "\n\n".join(blocks)


def _extract_status_code(exc: Exception) -> int | None:
    """Extract integer HTTP status code from an exception or its attributes."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    msg = str(exc)
    for candidate in [503, 502, 504, 500, 429, 404, 403, 401, 400]:
        if f"{candidate}" in msg:
            return candidate
    return None


def _is_retryable(exc: Exception) -> bool:
    """Determine whether an error is transient and safe to retry."""
    code = _extract_status_code(exc)
    if code in RETRYABLE_STATUS_CODES:
        return True
    if code in {400, 401, 403, 404}:
        return False
    exc_name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if any(term in exc_name or term in msg for term in ["network", "timeout", "connection", "connect", "timed out"]):
        return True
    return False


def _make_friendly_error(exc: Exception, model_name: str, attempts: int) -> LLMError:
    """Map raw exception to a user-friendly LLMError and log full trace."""
    logger.error("Gemini API error on model %s after %d attempts: %s", model_name, attempts, exc, exc_info=True)
    code = _extract_status_code(exc)

    if code == 503:
        return LLMError(f"Gemini is overloaded right now (tried {attempts} times). Wait a minute and try again.")
    elif code == 429:
        return LLMError("Gemini rate limit or quota reached. Wait a minute, or check your quota in Google AI Studio.")
    elif code in (401, 403):
        return LLMError("Gemini rejected the API key. Check GEMINI_API_KEY in .env.")
    elif code == 404:
        return LLMError(f"Model '{model_name}' was not found. Check GEMINI_MODEL in .env.")
    else:
        code_str = str(code) if code is not None else exc.__class__.__name__
        return LLMError(f"Gemini request failed ({code_str}). See the terminal log.")


def _read_cache(cache_path: str) -> dict[str, Any]:
    """Load cached Q&A dictionary from disk."""
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read LLM cache file %s: %s", cache_path, exc)
        return {}


def _write_cache(cache_path: str, key: str, value: dict[str, Any]) -> None:
    """Store cached Q&A dictionary to disk."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        data = _read_cache(cache_path)
        data[key] = value
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to write to LLM cache file %s: %s", cache_path, exc)


def _call_model(
    client: Any,
    model: str,
    prompt: str,
    generate_fn: Callable[..., str] | None = None,
    system_instruction: str | None = SYSTEM_PROMPT,
) -> str:
    """Invoke either the injectable generator or the Google GenAI SDK."""
    if generate_fn is not None:
        try:
            import inspect
            sig = inspect.signature(generate_fn)
            if len(sig.parameters) >= 2:
                return generate_fn(prompt, model)
            return generate_fn(prompt)
        except TypeError:
            return generate_fn(prompt)

    from google.genai import types

    cfg_args: dict[str, Any] = {
        "temperature": 0.2,
        "max_output_tokens": 512,
    }
    if system_instruction is not None:
        cfg_args["system_instruction"] = system_instruction

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**cfg_args),
    )
    if not response or not response.text:
        raise LLMError("Received an empty or blocked response from Gemini API.")
    return response.text.strip()


def answer_question(
    question: str,
    chunks: list[dict[str, Any]],
    generate_fn: Callable[..., str] | None = None,
    use_cache: bool = True,
    sleep_fn: Callable[[float], None] = time.sleep,
    cache_path: str = CACHE_FILE_DEFAULT,
) -> dict[str, Any]:
    """Generate a grounded answer for the question using retrieved chunks.

    Supports retries on transient errors, fallback model execution, disk caching,
    and user-friendly error wrapping.

    Args:
        question: The user's question.
        chunks: List of retrieved chunk dictionaries.
        generate_fn: Optional injectable generator for testing without API.
        use_cache: If True, reads and writes answers from cache/llm_cache.json.
        sleep_fn: Function used to pause between retries (defaults to time.sleep).
        cache_path: Path to disk cache JSON file.

    Returns:
        dict: {"answer": str, "grounded": bool, "cited": list[int], "sources": list[dict],
               "model": str, "latency_sec": float, "fallback_used": bool, "attempts": int,
               "cache_hit": bool}
    """
    load_dotenv()
    main_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "").strip() or None

    context_str = build_context(chunks)
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

    start_time = time.perf_counter()

    # Check cache
    concat_chunks = "".join(str(c.get("text", "")) for c in chunks)
    cache_key = hashlib.sha256(f"{main_model}|{question}|{concat_chunks}".encode("utf-8")).hexdigest()

    if use_cache:
        cached_dict = _read_cache(cache_path)
        if cache_key in cached_dict:
            cached_res = cached_dict[cache_key]
            return {
                "answer": cached_res["answer"],
                "grounded": cached_res["grounded"],
                "cited": cached_res["cited"],
                "sources": cached_res["sources"],
                "model": cached_res["model"],
                "latency_sec": round(time.perf_counter() - start_time, 2),
                "fallback_used": cached_res.get("fallback_used", False),
                "attempts": 0,
                "cache_hit": True,
            }

    client = None if generate_fn is not None else get_client()

    total_attempts = 0
    active_model = main_model
    fallback_used = False
    raw_answer: str | None = None
    last_exception: Exception | None = None

    # Main model: 1 try + up to 3 retries (delays: 2s, 5s, 10s)
    main_delays = [2.0, 5.0, 10.0]

    for attempt_idx in range(len(main_delays) + 1):
        total_attempts += 1
        try:
            raw_answer = _call_model(client, active_model, prompt, generate_fn).strip()
            last_exception = None
            break
        except Exception as exc:
            last_exception = exc
            if not _is_retryable(exc):
                raise _make_friendly_error(exc, active_model, total_attempts) from exc
            if attempt_idx < len(main_delays):
                delay = main_delays[attempt_idx] + random.uniform(0.1, 0.5)
                sleep_fn(delay)

    # If main model failed after all retries, attempt fallback model if configured
    if last_exception is not None and fallback_model and fallback_model != main_model:
        logger.info("Main model %s exhausted retries. Attempting fallback model %s.", main_model, fallback_model)
        fallback_used = True
        active_model = fallback_model
        fallback_delays = [2.0]

        for fb_idx in range(len(fallback_delays) + 1):
            total_attempts += 1
            try:
                raw_answer = _call_model(client, active_model, prompt, generate_fn).strip()
                last_exception = None
                break
            except Exception as exc:
                last_exception = exc
                if not _is_retryable(exc):
                    raise _make_friendly_error(exc, active_model, total_attempts) from exc
                if fb_idx < len(fallback_delays):
                    delay = fallback_delays[fb_idx] + random.uniform(0.1, 0.5)
                    sleep_fn(delay)

    if last_exception is not None:
        raise _make_friendly_error(last_exception, active_model, total_attempts) from last_exception

    assert raw_answer is not None
    latency_sec = round(time.perf_counter() - start_time, 2)

    # Check for ungrounded answer
    if "NOT_IN_VIDEO" in raw_answer:
        result = {
            "answer": raw_answer,
            "grounded": False,
            "cited": [],
            "sources": [],
            "model": active_model,
            "latency_sec": latency_sec,
            "fallback_used": fallback_used,
            "attempts": total_attempts,
            "cache_hit": False,
        }
        if use_cache:
            _write_cache(cache_path, cache_key, result)
        return result

    # Parse citations [n]
    citations = re.findall(r"\[(\d+)\]", raw_answer)
    cited_numbers: list[int] = []
    for c_str in citations:
        num = int(c_str)
        if 1 <= num <= len(chunks) and num not in cited_numbers:
            cited_numbers.append(num)

    sources: list[dict[str, Any]] = []
    for n in cited_numbers:
        chunk = chunks[n - 1]
        sources.append({
            "n": n,
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
            "text": chunk["text"],
        })

    # If answer is grounded but no valid citations were made, fall back to top-1 chunk
    if not sources and chunks:
        top_chunk = chunks[0]
        cited_numbers = [1]
        sources = [{
            "n": 1,
            "start": float(top_chunk["start"]),
            "end": float(top_chunk["end"]),
            "text": top_chunk["text"],
        }]

    result = {
        "answer": raw_answer,
        "grounded": True,
        "cited": cited_numbers,
        "sources": sources,
        "model": active_model,
        "latency_sec": latency_sec,
        "fallback_used": fallback_used,
        "attempts": total_attempts,
        "cache_hit": False,
    }

    if use_cache:
        _write_cache(cache_path, cache_key, result)

    return result


def log_qa(
    video_id: str,
    strategy: str,
    k: int,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    llm_result: dict[str, Any],
    log_path: str = "logs/qa_log.jsonl",
) -> None:
    """Append Q&A record to logs/qa_log.jsonl for evaluation and tracking."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id,
        "strategy": strategy,
        "k": k,
        "question": question,
        "retrieved": [
            {
                "chunk_id": c.get("chunk_id"),
                "score": round(float(c.get("score", 0.0)), 4),
            }
            for c in retrieved_chunks
        ],
        "cited": llm_result.get("cited", []),
        "grounded": llm_result.get("grounded", False),
        "answer": llm_result.get("answer", ""),
        "model": llm_result.get("model", ""),
        "fallback_used": llm_result.get("fallback_used", False),
        "attempts": llm_result.get("attempts", 1),
        "cache_hit": llm_result.get("cache_hit", False),
        "latency_sec": llm_result.get("latency_sec", 0.0),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
