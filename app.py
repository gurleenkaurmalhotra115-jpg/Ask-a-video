"""Ask a Video (VidQuery) - Streamlit Application."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
import os
import pandas as pd
import streamlit as st

from chunking import DEFAULT_PARAMS, STRATEGIES, chunk_stats, chunk_transcript
from llm import LLMError, answer_question, log_qa
from transcript import TranscriptError, fmt_time, get_transcript
from vectorstore import get_client, index_chunks, list_indexed, search


@st.cache_resource
def get_vectorstore_client():
    """Lazily initialize and cache the local Qdrant client singleton."""
    return get_client("qdrant_data")


@st.cache_data(show_spinner=False)
def get_cached_chunks(
    video_id: str,
    segments: list[dict[str, Any]],
    strategy: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute and cache chunks keyed on video id, strategy, and parameter configuration."""
    return chunk_transcript(segments, strategy, **params)


@st.cache_data(show_spinner=False)
def get_suggested_questions(video_id: str, segments: list[dict[str, Any]]) -> list[str]:
    """Retrieve 3 pedagogical questions spanning early, middle, and late sections of the video."""
    questions_csv = "eval/questions.csv"
    if os.path.exists(questions_csv):
        try:
            df_q = pd.read_csv(questions_csv)
            v_matches = df_q[(df_q["video_id"] == video_id) & (df_q["is_negative"] == False)]
            if not v_matches.empty:
                sorted_matches = v_matches.sort_values(by="gold_start_sec")
                n = len(sorted_matches)
                if n <= 3:
                    return sorted_matches["question"].tolist()
                indices = [min(1, n - 1), n // 2, n - 1]
                seen = set()
                res = []
                for idx in indices:
                    q = str(sorted_matches.iloc[idx]["question"]).strip()
                    if q not in seen:
                        seen.add(q)
                        res.append(q)
                return res
        except Exception:
            pass

    if not segments:
        return [
            "What is the main topic covered in this video?",
            "What key methods or techniques are introduced?",
            "What conclusions or summaries are discussed at the end?",
        ]

    n_seg = len(segments)
    p1 = segments[min(5, n_seg - 1)]["text"]
    p2 = segments[n_seg // 2]["text"]
    p3 = segments[max(0, n_seg - 6)]["text"]

    def _truncate(t: str, max_w: int = 6) -> str:
        words = t.split()
        return " ".join(words[:max_w])

    return [
        f"What is introduced regarding '{_truncate(p1)}'?",
        f"How does the lecture explain '{_truncate(p2)}'?",
        f"What is summarized near the end about '{_truncate(p3)}'?",
    ]


def generate_study_note_markdown(
    video_id: str,
    question: str,
    qa_res: dict[str, Any],
) -> str:
    """Generate a clean, structured Markdown study note with cited timestamps."""
    yt_url = f"https://www.youtube.com/watch?v={video_id}"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# 🎓 VidQuery Grounded Study Note",
        f"",
        f"- **Lecture URL:** [{yt_url}]({yt_url})",
        f"- **Question:** {question}",
        f"- **Grounded LLM:** `{qa_res.get('model', 'gemini-3.5-flash-lite')}`",
        f"- **Latency:** `{qa_res.get('latency_sec', 0.0)}s` | **Generated:** {now_str}",
        f"",
        f"---",
        f"",
        f"## 📝 Grounded Answer",
        f"",
        f"{qa_res.get('answer', '')}",
        f"",
        f"---",
        f"",
        f"## ⏱️ Verified Timestamp Citations & Excerpts",
        f"",
    ]
    sources = qa_res.get("sources", [])
    if not sources:
        lines.append("*No citations provided for this query.*")
    for src in sources:
        t_start = fmt_time(src["start"])
        t_end = fmt_time(src["end"])
        seek_s = int(round(src["start"]))
        direct_link = f"https://www.youtube.com/watch?v={video_id}&t={seek_s}s"
        lines.append(f"### Citation [{src['n']}] — `{t_start} - {t_end}`")
        lines.append(f"> \"{src['text']}\"")
        lines.append(f"")
        lines.append(f"👉 [Seek Video to {t_start}]({direct_link})")
        lines.append(f"")

    lines.append("---")
    lines.append("*Generated automatically by VidQuery — Timestamp-Grounded Video RAG*")
    return "\n".join(lines)


def is_seg_cited(seg_start: float, seg_dur: float, spans: list[tuple[float, float, int]]) -> int | None:
    """Check if a transcript segment overlaps with any cited source span."""
    seg_end = seg_start + max(1.0, seg_dur)
    for start, end, n in spans:
        if max(seg_start, start) < min(seg_end, end):
            return n
    return None


def main() -> None:
    """Run the main Streamlit application interface."""
    st.set_page_config(page_title="Ask a Video (VidQuery)", page_icon="🎬", layout="wide")

    st.markdown(
        """
        <style>
        /* Modern Clean White Theme */
        .stApp {
            background-color: #ffffff;
            color: #0f172a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        /* Top Navigation Header */
        .app-header {
            margin-bottom: 1.2rem;
            padding-bottom: 0.8rem;
            border-bottom: 1px solid #e2e8f0;
        }
        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.025em;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .app-badge {
            font-size: 0.85rem;
            font-weight: 600;
            color: #2563eb;
            background: #eff6ff;
            padding: 3px 12px;
            border-radius: 12px;
            border: 1px solid #bfdbfe;
        }
        .app-subtitle {
            font-size: 0.95rem;
            color: #64748b;
            margin: 0.25rem 0 0 0;
        }
        /* Buttons */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.15s ease-in-out !important;
            border: 1px solid #cbd5e1 !important;
            background-color: #ffffff !important;
            color: #1e293b !important;
        }
        .stButton > button:hover {
            border-color: #2563eb !important;
            color: #2563eb !important;
            background-color: #eff6ff !important;
            box-shadow: 0 1px 3px rgba(37, 99, 235, 0.12) !important;
        }
        /* Primary buttons */
        .stButton > button[kind="primary"] {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border: 1px solid #1d4ed8 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #1d4ed8 !important;
            color: #ffffff !important;
        }
        /* Text inputs */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: 8px !important;
            border: 1px solid #cbd5e1 !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
        }
        /* Metrics & Containers */
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 12px;
            border-radius: 8px;
        }
        .answer-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 18px 20px;
            margin-top: 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        </style>
        <div class="app-header">
            <div class="app-title">🎬 Ask a Video <span class="app-badge">VidQuery RAG</span></div>
            <p class="app-subtitle">Timestamp-grounded educational lecture Q&amp;A powered by dense retrieval &amp; Gemini</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state storage
    if "videos" not in st.session_state:
        st.session_state.videos = {}
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None
    if "seek" not in st.session_state:
        st.session_state.seek = 0
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False
    if "latest_qa" not in st.session_state:
        st.session_state.latest_qa = None
    if "last_llm_error" not in st.session_state:
        st.session_state.last_llm_error = None
    if "last_error_retrieved" not in st.session_state:
        st.session_state.last_error_retrieved = None
    if "current_question_text" not in st.session_state:
        st.session_state.current_question_text = ""
    if "trigger_ask" not in st.session_state:
        st.session_state.trigger_ask = False

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        use_whisper = st.checkbox(
            "Use Whisper if no captions (slow)",
            value=False,
            help="Attempt local Whisper transcription if official captions are not found.",
        )

        st.subheader("Chunking Strategy")
        strategy_options = list(STRATEGIES.keys())
        selected_strategy = st.selectbox(
            "Select strategy",
            strategy_options,
            index=0,
            help="Choose the chunking algorithm used to group transcript segments.",
        )

        strategy_params: dict[str, Any] = {}
        if selected_strategy == "fixed_token":
            c_words = st.slider(
                "Chunk words (window size)",
                min_value=20,
                max_value=500,
                value=int(DEFAULT_PARAMS["fixed_token"]["chunk_words"]),
                step=10,
            )
            o_words = st.slider(
                "Overlap words",
                min_value=0,
                max_value=100,
                value=int(DEFAULT_PARAMS["fixed_token"]["overlap_words"]),
                step=5,
            )
            strategy_params = {"chunk_words": c_words, "overlap_words": o_words}

        elif selected_strategy == "sentence":
            s_per_chunk = st.slider(
                "Sentences per chunk",
                min_value=1,
                max_value=20,
                value=int(DEFAULT_PARAMS["sentence"]["sentences_per_chunk"]),
                step=1,
            )
            o_sents = st.slider(
                "Overlap sentences",
                min_value=0,
                max_value=5,
                value=int(DEFAULT_PARAMS["sentence"]["overlap_sentences"]),
                step=1,
            )
            strategy_params = {"sentences_per_chunk": s_per_chunk, "overlap_sentences": o_sents}

        elif selected_strategy == "semantic":
            pct = st.slider(
                "Similarity drop percentile",
                min_value=5,
                max_value=95,
                value=int(DEFAULT_PARAMS["semantic"]["percentile"]),
                step=5,
            )
            min_w = st.slider(
                "Min words per chunk",
                min_value=10,
                max_value=100,
                value=int(DEFAULT_PARAMS["semantic"]["min_words"]),
                step=5,
            )
            max_w = st.slider(
                "Max words per chunk",
                min_value=50,
                max_value=500,
                value=int(DEFAULT_PARAMS["semantic"]["max_words"]),
                step=10,
            )
            strategy_params = {"percentile": pct, "min_words": min_w, "max_words": max_w}

        elif selected_strategy == "time_window":
            w_sec = st.slider(
                "Window duration (sec)",
                min_value=10.0,
                max_value=180.0,
                value=float(DEFAULT_PARAMS["time_window"]["window_sec"]),
                step=5.0,
            )
            o_sec = st.slider(
                "Overlap duration (sec)",
                min_value=0.0,
                max_value=60.0,
                value=float(DEFAULT_PARAMS["time_window"]["overlap_sec"]),
                step=2.0,
            )
            strategy_params = {"window_sec": w_sec, "overlap_sec": o_sec}

    # Main input controls
    CACHED_BENCHMARK_VIDEOS = {
        "-- Select a benchmark video (Instant offline) --": "",
        "3Blue1Brown - What is a Neural Network? (aircAruvnKk)": "https://www.youtube.com/watch?v=aircAruvnKk",
        "Andrej Karpathy - Intro to Large Language Models (zjkBMFhNj_g)": "https://www.youtube.com/watch?v=zjkBMFhNj_g",
        "StatQuest - Transformer Neural Networks (zxQyTK8quyY)": "https://www.youtube.com/watch?v=zxQyTK8quyY",
        "StatQuest - Neural Network Fundamentals (CqOfi41LfDw)": "https://www.youtube.com/watch?v=CqOfi41LfDw",
        "Fast.ai - Practical Deep Learning Lesson 1 (8SF_h3xF3cE)": "https://www.youtube.com/watch?v=8SF_h3xF3cE",
        "Stanford CS229 - ML Lecture 1 by Andrew Ng (jGwO_UgTS7I)": "https://www.youtube.com/watch?v=jGwO_UgTS7I",
        "MIT 6.S191 - Introduction to Deep Learning (II4giR4vOOo)": "https://www.youtube.com/watch?v=II4giR4vOOo",
        "MIT Strang - Geometry of Linear Equations (J7DzL2_Na80)": "https://www.youtube.com/watch?v=J7DzL2_Na80",
        "Andrej Karpathy - Building makemore (PaCmpygFfXo)": "https://www.youtube.com/watch?v=PaCmpygFfXo",
        "Andrew Ng - Opportunities in AI 2023 (5p248yoa3oE)": "https://www.youtube.com/watch?v=5p248yoa3oE",
        "NPTEL IIT Madras - Intro to Machine Learning (YLczBemZpaE)": "https://www.youtube.com/watch?v=YLczBemZpaE",
        "NPTEL IIT Madras - Artificial Intelligence (zT0sGjOSPt8)": "https://www.youtube.com/watch?v=zT0sGjOSPt8",
        "Krish Naik - End To End Machine Learning Project (MJ1vWb1rGwM)": "https://www.youtube.com/watch?v=MJ1vWb1rGwM",
    }

    def on_benchmark_change() -> None:
        chosen = st.session_state.get("selected_demo_key", "")
        url_match = CACHED_BENCHMARK_VIDEOS.get(chosen, "")
        if url_match:
            st.session_state["video_url_text"] = url_match

    if "video_url_text" not in st.session_state:
        st.session_state.video_url_text = ""

    selected_demo = st.selectbox(
        "⚡ Choose a pre-indexed benchmark lecture (Instant offline, no YouTube rate limits):",
        options=list(CACHED_BENCHMARK_VIDEOS.keys()),
        index=0,
        key="selected_demo_key",
        on_change=on_benchmark_change,
        help="Select any of the 13 benchmark lecture videos to load its transcript instantly from disk without hitting YouTube API.",
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        url_input = st.text_input(
            "YouTube Video URL or ID",
            key="video_url_text",
            placeholder="e.g. https://www.youtube.com/watch?v=... or youtu.be/...",
            label_visibility="collapsed",
        )
    with col_btn:
        fetch_clicked = st.button("Fetch transcript", type="primary", use_container_width=True)

    if fetch_clicked:
        clean_url = url_input.strip()
        if not clean_url:
            st.warning("Please enter a YouTube video URL or ID.")
        else:
            try:
                with st.spinner("Fetching transcript..."):
                    video_id, segments, source = get_transcript(
                        clean_url, allow_whisper=use_whisper
                    )
                    st.session_state.videos[video_id] = {
                        "segments": segments,
                        "source": source,
                    }
                    st.session_state.current_video_id = video_id
                    st.session_state.seek = 0
                    st.session_state.autoplay = False
                    st.session_state.latest_qa = None
                    st.session_state.last_llm_error = None
                    st.session_state.last_error_retrieved = None
                    st.session_state.current_question_text = ""
            except (TranscriptError, LLMError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"An unexpected error occurred: {exc}")

    # Display results when a video is loaded
    current_video_id = st.session_state.get("current_video_id")
    if current_video_id and current_video_id in st.session_state.videos:
        video_data = st.session_state.videos[current_video_id]
        segments = video_data["segments"]
        source = video_data["source"]
        youtube_url = f"https://www.youtube.com/watch?v={current_video_id}"

        total_duration = 0.0
        if segments:
            last_seg = segments[-1]
            total_duration = float(last_seg["start"]) + float(last_seg["duration"])
        duration_formatted = fmt_time(total_duration)

        st.divider()
        col_video, col_main = st.columns([1, 1])

        with col_video:
            st.subheader("Video")
            seek_val = st.session_state.get("seek", 0)
            autoplay_val = st.session_state.get("autoplay", False)
            st.video(youtube_url, start_time=seek_val, autoplay=autoplay_val)

        with col_main:
            st.success(
                f"Retrieved {len(segments)} segments (total duration: {duration_formatted}) via **{source}**."
            )

            # Six views: Ask, Transcript, Chunks, Retrieval test, Strategy comparison, Benchmark & Paper
            tab_ask, tab_transcript, tab_chunks, tab_retrieval, tab_comparison, tab_results = st.tabs(
                [
                    "Ask",
                    "Transcript",
                    "Chunks",
                    "Retrieval test",
                    "Strategy comparison",
                    "📊 Benchmark & Paper",
                ]
            )

            # -------------------- TAB 1: ASK --------------------
            with tab_ask:
                st.subheader("Ask a Question")
                client = get_vectorstore_client()
                indexed_items = list_indexed(client)
                indexed_strategies = {
                    item["strategy"]
                    for item in indexed_items
                    if item["video_id"] == current_video_id
                }

                col_ask_strat, col_ask_k = st.columns([2, 1])
                with col_ask_strat:
                    strat_keys = list(DEFAULT_PARAMS.keys())
                    default_strat_idx = (
                        strat_keys.index(selected_strategy)
                        if selected_strategy in strat_keys
                        else 0
                    )
                    ask_strategy = st.selectbox(
                        "Strategy for retrieval",
                        options=strat_keys,
                        index=default_strat_idx,
                        key="ask_strategy_select",
                    )
                with col_ask_k:
                    ask_k = st.slider(
                        "Top-k chunks", min_value=1, max_value=10, value=4, key="ask_k_slider"
                    )

                # Dynamic Suggested Question Chips
                suggested_qs = get_suggested_questions(current_video_id, segments)
                if suggested_qs:
                    st.markdown("**💡 Suggested Questions from this Lecture:**")
                    cols_sug = st.columns(len(suggested_qs))
                    for i_s, q_s in enumerate(suggested_qs):
                        with cols_sug[i_s]:
                            short_label = q_s if len(q_s) <= 42 else q_s[:39] + "..."
                            def make_sug_callback(q_target: str):
                                def _callback():
                                    st.session_state["ask_question_input"] = q_target
                                    st.session_state["current_question_text"] = q_target
                                    st.session_state["trigger_ask"] = True
                                return _callback

                            st.button(
                                f"📌 {short_label}",
                                key=f"btn_sug_q_{i_s}",
                                help=q_s,
                                use_container_width=True,
                                on_click=make_sug_callback(q_s),
                            )

                question_text = st.text_area(
                    "Your question about this video:",
                    placeholder="e.g. What is the difference between supervised and unsupervised learning?",
                    key="ask_question_input",
                )

                col_btn_ask, col_btn_clear = st.columns([4, 1])
                with col_btn_ask:
                    ask_btn = st.button("Ask Question", type="primary", use_container_width=True)
                with col_btn_clear:
                    def clear_btn_callback():
                        st.session_state["ask_question_input"] = ""
                        st.session_state["current_question_text"] = ""
                        st.session_state["latest_qa"] = None
                        st.session_state["last_llm_error"] = None
                        st.session_state["trigger_ask"] = False

                    st.button("Clear", use_container_width=True, on_click=clear_btn_callback)

                trigger_exec = ask_btn or st.session_state.get("trigger_ask", False)
                if trigger_exec:
                    st.session_state.trigger_ask = False
                    clean_q = (
                        question_text.strip()
                        or st.session_state.get("current_question_text", "").strip()
                    )
                    if not clean_q:
                        st.warning("Please enter a question or click a suggested question chip above.")
                    else:
                        st.session_state.current_question_text = clean_q
                        st.session_state.last_error_retrieved = None
                        st.session_state.last_llm_error = None

                        # Auto-index on the fly if this strategy is not yet indexed in Qdrant
                        if ask_strategy not in indexed_strategies:
                            with st.spinner(f"Auto-indexing '{ask_strategy}' chunks for first-time search..."):
                                chunks_to_index = get_cached_chunks(
                                    current_video_id,
                                    segments,
                                    ask_strategy,
                                    DEFAULT_PARAMS[ask_strategy],
                                )
                                index_chunks(client, current_video_id, ask_strategy, chunks_to_index)
                                indexed_strategies.add(ask_strategy)

                        try:
                            with st.spinner("Searching video & generating answer..."):
                                retrieved = search(
                                    client=client,
                                    query=clean_q,
                                    video_id=current_video_id,
                                    strategy=ask_strategy,
                                    k=ask_k,
                                )
                                if not retrieved:
                                    st.warning("No relevant chunks found in the video.")
                                else:
                                    try:
                                        qa_result = answer_question(
                                            question=clean_q, chunks=retrieved
                                        )
                                        log_qa(
                                            video_id=current_video_id,
                                            strategy=ask_strategy,
                                            k=ask_k,
                                            question=clean_q,
                                            retrieved_chunks=retrieved,
                                            llm_result=qa_result,
                                        )
                                        st.session_state.latest_qa = qa_result
                                        st.session_state.last_llm_error = None
                                    except LLMError as exc:
                                        st.session_state.last_llm_error = str(exc)
                                        st.session_state.last_error_retrieved = retrieved
                                        st.session_state.latest_qa = None
                        except TranscriptError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Error generating answer: {exc}")

                # Display error and fallback retrieved chunks if LLM was unavailable
                if st.session_state.get("last_llm_error"):
                    st.error(st.session_state["last_llm_error"])
                    err_chunks = st.session_state.get("last_error_retrieved", [])
                    if err_chunks:
                        st.markdown("### Best matching moments (Gemini unavailable)")
                        for s_idx, src in enumerate(err_chunks):
                            col_b, col_t = st.columns([1, 4])
                            with col_b:
                                time_label = f"▶ {fmt_time(src['start'])}"
                                if st.button(
                                    time_label,
                                    key=f"err_src_{src.get('chunk_id', s_idx)}_{s_idx}",
                                    use_container_width=True,
                                ):
                                    st.session_state.seek = int(round(src["start"]))
                                    st.session_state.autoplay = True
                                    st.rerun()
                            with col_t:
                                st.markdown(
                                    f"**[{s_idx + 1}]** *({fmt_time(src['start'])}-{fmt_time(src['end'])})* {src['text']}"
                                )

                # Display the latest QA result
                qa_res = st.session_state.get("latest_qa")
                if qa_res:
                    st.markdown("---")
                    if not qa_res.get("grounded", False):
                        st.warning("⚠️ This video does not answer this question (Out-of-Domain Negative Rejection).")
                    else:
                        st.markdown(
                            f"""<div class="answer-card">
                            <h3 style="margin-top:0; color:#0f172a; font-size:1.15rem; font-weight:700;">💬 Grounded Answer</h3>
                            <div style="font-size:1rem; line-height:1.65; color:#1e293b;">{qa_res['answer']}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        model_parts = [f"Model: `{qa_res.get('model', 'gemini-3.8-flash')}`"]
                        if qa_res.get("fallback_used"):
                            model_parts.append("*(fallback)*")
                        if qa_res.get("cache_hit"):
                            model_parts.append("*(cached)*")
                        model_parts.append(f"| Latency: `{qa_res.get('latency_sec', 0.0)}s`")
                        st.caption(" ".join(model_parts))

                        sources = qa_res.get("sources", [])
                        if sources:
                            st.markdown("### Sources")
                            for s_idx, src in enumerate(sources):
                                col_b, col_t = st.columns([1, 4])
                                with col_b:
                                    time_label = f"▶ {fmt_time(src['start'])}"
                                    if st.button(
                                        time_label,
                                        key=f"ask_src_{src['n']}_{s_idx}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.seek = int(round(src["start"]))
                                        st.session_state.autoplay = True
                                        st.rerun()
                                with col_t:
                                    yt_direct_url = f"https://www.youtube.com/watch?v={current_video_id}&t={int(round(src['start']))}s"
                                    st.markdown(
                                        f"**[{src['n']}]** *({fmt_time(src['start'])}-{fmt_time(src['end'])})* {src['text']} "
                                        f"[[Open in YouTube ↗]({yt_direct_url})]"
                                    )

                        # Feature 3: Export Study Notes Button
                        if qa_res.get("grounded", False):
                            st.markdown("---")
                            study_note_md = generate_study_note_markdown(
                                video_id=current_video_id,
                                question=st.session_state.get("current_question_text", ""),
                                qa_res=qa_res,
                            )
                            st.download_button(
                                label="📥 Export Grounded Study Note (.md)",
                                data=study_note_md,
                                file_name=f"vidquery_study_note_{current_video_id}.md",
                                mime="text/markdown",
                                use_container_width=True,
                                help="Download your Q&A, verified citations, and timestamp links as an exam-ready Markdown study note.",
                            )

            # -------------------- TAB 2: TRANSCRIPT --------------------
            with tab_transcript:
                st.subheader("Interactive Transcript")

                active_qa = st.session_state.get("latest_qa")
                active_sources = active_qa.get("sources", []) if active_qa else []
                cited_spans = [
                    (float(s["start"]), float(s["end"]), int(s.get("n", 1)))
                    for s in active_sources
                ]

                col_t_search, col_t_mode = st.columns([3, 2])
                with col_t_search:
                    filter_kw = st.text_input(
                        "🔍 Search transcript by keyword:",
                        "",
                        key="transcript_filter_kw",
                        placeholder="e.g. gradient, neuron, attention",
                    ).strip().lower()
                with col_t_mode:
                    view_mode = st.radio(
                        "Display format:",
                        ["Interactive (Click-to-Seek)", "Data Table"],
                        horizontal=True,
                        key="transcript_view_mode",
                    )

                if active_sources:
                    st.info(
                        f"🎯 **{len(active_sources)} cited moment(s)** from your latest answer are highlighted in yellow below."
                    )
                    only_cited = st.checkbox(
                        "Show only cited segments",
                        value=False,
                        key="chk_only_cited",
                    )
                else:
                    only_cited = False

                if view_mode == "Data Table":
                    df_records = []
                    for seg in segments:
                        c_n = is_seg_cited(seg["start"], seg["duration"], cited_spans)
                        if only_cited and c_n is None:
                            continue
                        if filter_kw and filter_kw not in seg["text"].lower():
                            continue
                        df_records.append({
                            "time (mm:ss)": fmt_time(seg["start"]),
                            "start (s)": seg["start"],
                            "cited": f"Source [{c_n}]" if c_n else "",
                            "text": seg["text"],
                        })
                    st.dataframe(pd.DataFrame(df_records), use_container_width=True, height=450)
                else:
                    filtered_segs = []
                    for seg in segments:
                        c_n = is_seg_cited(seg["start"], seg["duration"], cited_spans)
                        if only_cited and c_n is None:
                            continue
                        if filter_kw and filter_kw not in seg["text"].lower():
                            continue
                        filtered_segs.append((seg, c_n))

                    st.caption(
                        f"Showing {len(filtered_segs)} of {len(segments)} segments. Click any timestamp to seek the video."
                    )

                    with st.container(height=520):
                        for idx, (seg, c_n) in enumerate(filtered_segs):
                            col_btn_seek, col_text_seg = st.columns([1, 5])
                            with col_btn_seek:
                                if st.button(
                                    f"▶ {fmt_time(seg['start'])}",
                                    key=f"seek_tr_{idx}_{int(seg['start'])}",
                                    use_container_width=True,
                                ):
                                    st.session_state.seek = int(round(seg["start"]))
                                    st.session_state.autoplay = True
                                    st.rerun()
                            with col_text_seg:
                                if c_n is not None:
                                    st.markdown(
                                        f"""<div style="background-color: #fef3c7; border-left: 4px solid #d97706; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px;">
                                        <span style="color: #b45309; font-weight: 700; font-size: 0.82em;">🎯 CITED AS SOURCE [{c_n}]</span><br/>
                                        <span style="color: #451a03; font-size: 0.95rem; line-height: 1.5;">{seg['text']}</span>
                                        </div>""",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(f"<span style='color: #64748b; font-weight: 600;'>{fmt_time(seg['start'])}</span> — <span style='color: #1e293b;'>{seg['text']}</span>", unsafe_allow_html=True)

            # -------------------- TAB 3: CHUNKS --------------------
            with tab_chunks:
                spinner_msg = (
                    "Computing semantic embeddings & chunks (first run downloads model)..."
                    if selected_strategy == "semantic"
                    else f"Computing '{selected_strategy}' chunks..."
                )
                with st.spinner(spinner_msg):
                    active_chunks = get_cached_chunks(
                        current_video_id, segments, selected_strategy, strategy_params
                    )
                    stats = chunk_stats(active_chunks)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Chunks", stats["n_chunks"])
                m2.metric("Avg Words", stats["avg_words"])
                m3.metric("Avg Duration", f"{stats['avg_duration_sec']}s")
                m4.metric("Duration Range", f"{stats['min_duration']}s - {stats['max_duration']}s")

                if stats["avg_words"] > 180:
                    st.warning("Chunks this long get truncated by the embedding model.")

                chunks_records = [
                    {
                        "chunk_id": c["chunk_id"],
                        "time range": f"{fmt_time(c['start'])} - {fmt_time(c['end'])}",
                        "duration (s)": round(c["end"] - c["start"], 2),
                        "n_words": c["n_words"],
                        "text": c["text"],
                    }
                    for c in active_chunks
                ]
                df_chunks = pd.DataFrame(chunks_records)
                st.dataframe(df_chunks, use_container_width=True, height=350)

                st.markdown("---")
                st.subheader("Vector Indexing")
                if st.button("Index all 4 strategies", type="primary", use_container_width=True):
                    client = get_vectorstore_client()
                    progress_bar = st.progress(0.0)
                    summary_rows = []
                    strat_keys = list(DEFAULT_PARAMS.keys())

                    for idx, strat in enumerate(strat_keys):
                        spinner_text = (
                            "Computing & indexing semantic embeddings (first run downloads model)..."
                            if strat == "semantic"
                            else f"Indexing '{strat}' chunks..."
                        )
                        with st.spinner(spinner_text):
                            chunks_to_index = get_cached_chunks(
                                current_video_id, segments, strat, DEFAULT_PARAMS[strat]
                            )
                            n_indexed = index_chunks(client, current_video_id, strat, chunks_to_index)
                            summary_rows.append({
                                "Strategy": strat,
                                "Chunks Indexed": n_indexed,
                            })
                        progress_bar.progress((idx + 1) / len(strat_keys))

                    st.success("Successfully indexed all 4 strategies in Qdrant!")
                    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            # -------------------- TAB 4: RETRIEVAL TEST --------------------
            with tab_retrieval:
                st.subheader("Retrieval Test")
                client = get_vectorstore_client()
                indexed_items = list_indexed(client)
                indexed_strategies = {
                    item["strategy"]
                    for item in indexed_items
                    if item["video_id"] == current_video_id
                }

                if not indexed_strategies:
                    st.info(
                        "This video has not been indexed yet. "
                        "Go to the **Chunks** tab and click **'Index all 4 strategies'** to index it."
                    )
                else:
                    col_q, col_s, col_k = st.columns([3, 1.5, 1.5])
                    with col_q:
                        query_input = st.text_input(
                            "Question / Query",
                            placeholder="e.g. What is gradient descent?",
                            key="query_input_field",
                        )
                    with col_s:
                        search_strategy = st.selectbox(
                            "Strategy",
                            options=list(DEFAULT_PARAMS.keys()),
                            index=0,
                            key="query_strat_select",
                        )
                    with col_k:
                        search_k = st.slider(
                            "Top-k results",
                            min_value=1,
                            max_value=10,
                            value=4,
                            key="query_k_slider",
                        )

                    if search_strategy not in indexed_strategies:
                        st.info(
                            f"Strategy '{search_strategy}' is not yet indexed for this video. "
                            "Click **'Index all 4 strategies'** in the **Chunks** tab."
                        )
                    elif query_input.strip():
                        with st.spinner("Searching Qdrant vector store..."):
                            search_results = search(
                                client=client,
                                query=query_input.strip(),
                                video_id=current_video_id,
                                strategy=search_strategy,
                                k=search_k,
                            )

                        if not search_results:
                            st.info("No matching chunks found.")
                        else:
                            st.write(f"Top {len(search_results)} retrieved chunks:")
                            for rank, res in enumerate(search_results, 1):
                                time_range = f"{fmt_time(res['start'])} - {fmt_time(res['end'])}"
                                with st.container():
                                    st.markdown(
                                        f"**Rank #{rank}** | Score: `{res['score']:.4f}` | ⏱️ `{time_range}` (Chunk ID: `{res['chunk_id']}`)"
                                    )
                                    st.info(res["text"])

                            top_start_sec = int(round(search_results[0]["start"]))
                            st.markdown("---")
                            st.subheader(
                                f"Playback Verification (top result at {fmt_time(top_start_sec)})"
                            )
                            st.video(youtube_url, start_time=top_start_sec)

            # -------------------- TAB 5: STRATEGY COMPARISON --------------------
            with tab_comparison:
                st.write("Comparison of all 4 chunking strategies using default parameters:")
                comparison_rows = []
                for strat_name, default_p in DEFAULT_PARAMS.items():
                    s_msg = (
                        "Evaluating semantic strategy..."
                        if strat_name == "semantic"
                        else f"Evaluating {strat_name}..."
                    )
                    with st.spinner(s_msg):
                        c_list = get_cached_chunks(
                            current_video_id, segments, strat_name, default_p
                        )
                        c_stats = chunk_stats(c_list)
                        comparison_rows.append({
                            "Strategy": strat_name,
                            "Total Chunks": c_stats["n_chunks"],
                            "Avg Words": c_stats["avg_words"],
                            "Avg Duration (s)": c_stats["avg_duration_sec"],
                            "Min Duration (s)": c_stats["min_duration"],
                            "Max Duration (s)": c_stats["max_duration"],
                        })
                df_comp = pd.DataFrame(comparison_rows)
                st.dataframe(df_comp, use_container_width=True)

            # -------------------- TAB 6: BENCHMARK & RESEARCH RESULTS --------------------
            with tab_results:
                st.subheader("📊 Empirical Research Findings & Benchmark Results")
                st.markdown(
                    "This tab summarizes the empirical findings from our IEEE conference paper: "
                    "*'Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise'*."
                )

                col_dl_pdf, col_dl_zip = st.columns([1, 1])
                with col_dl_pdf:
                    pdf_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "paper.pdf"
                    )
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 Download IEEE Research Paper (PDF)",
                                data=f.read(),
                                file_name="VidQuery_Research_Paper.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                with col_dl_zip:
                    zip_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "VidQuery_Submission.zip",
                    )
                    if os.path.exists(zip_path):
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                "📦 Download Full Submission Archive (.zip)",
                                data=f.read(),
                                file_name="VidQuery_Submission.zip",
                                mime="application/zip",
                                use_container_width=True,
                            )

                st.markdown("---")
                st.markdown("#### Table 1: Retrieval Benchmark Across Chunking Strategies (N=156)")
                csv_ret = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "results",
                    "retrieval_summary.csv",
                )
                if os.path.exists(csv_ret):
                    df_r = pd.read_csv(csv_ret)
                    st.dataframe(df_r, use_container_width=True)

                st.markdown("#### Table 2: Performance Breakdown by Pedagogical & Acoustic Tier")
                csv_t2 = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "results",
                    "tier_breakdown.csv",
                )
                if os.path.exists(csv_t2):
                    df_t2 = pd.read_csv(csv_t2)
                    st.dataframe(df_t2, use_container_width=True)

                st.markdown("#### Table 3: Video Duration & Search Space Expansion Effect")
                csv_t3 = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "results",
                    "duration_breakdown.csv",
                )
                if os.path.exists(csv_t3):
                    df_t3 = pd.read_csv(csv_t3)
                    st.dataframe(df_t3, use_container_width=True)

                st.markdown("#### Table 4: Generation Faithfulness & Zero Hallucination")
                csv_gen = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "results",
                    "generation_eval_summary.csv",
                )
                if os.path.exists(csv_gen):
                    df_g = pd.read_csv(csv_gen)
                    st.dataframe(df_g, use_container_width=True)

                st.info(
                    "💡 Slide presentation available locally: open `presentation.html` in your browser for the full 12-slide deck with speaker notes."
                )


if __name__ == "__main__":
    main()
