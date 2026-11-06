"""Ask a Video (VidQuery) - Streamlit Application."""

from __future__ import annotations

from typing import Any
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


def main() -> None:
    """Run the main Streamlit application interface."""
    st.set_page_config(page_title="Ask a Video", layout="wide")
    st.title("Ask a Video")

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

    selected_demo = st.selectbox(
        "⚡ Choose a pre-indexed benchmark lecture (Instant offline, no YouTube rate limits):",
        options=list(CACHED_BENCHMARK_VIDEOS.keys()),
        index=0,
        help="Select any of the 13 benchmark lecture videos to load its transcript instantly from disk without hitting YouTube API.",
    )
    default_url_val = CACHED_BENCHMARK_VIDEOS.get(selected_demo, "")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        url_input = st.text_input(
            "YouTube Video URL or ID",
            value=default_url_val,
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

            # Five views: Ask, Transcript, Chunks, Retrieval test, Strategy comparison
            tab_ask, tab_transcript, tab_chunks, tab_retrieval, tab_comparison = st.tabs(
                ["Ask", "Transcript", "Chunks", "Retrieval test", "Strategy comparison"]
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

                question_text = st.text_area(
                    "Your question about this video:",
                    value=st.session_state.get("current_question_text", ""),
                    placeholder="e.g. What is the difference between supervised and unsupervised learning?",
                    key="ask_question_input",
                )

                if ask_strategy not in indexed_strategies:
                    st.info(
                        f"Strategy '{ask_strategy}' is not indexed for this video yet."
                    )
                    if st.button(
                        f"Index '{ask_strategy}' now",
                        key="btn_index_for_ask",
                        use_container_width=True,
                    ):
                        with st.spinner(f"Indexing '{ask_strategy}' chunks..."):
                            chunks_to_index = get_cached_chunks(
                                current_video_id,
                                segments,
                                ask_strategy,
                                DEFAULT_PARAMS[ask_strategy],
                            )
                            index_chunks(client, current_video_id, ask_strategy, chunks_to_index)
                            st.success(f"Indexed {len(chunks_to_index)} chunks for '{ask_strategy}'!")
                            st.rerun()
                else:
                    ask_btn = st.button("Ask", type="primary", use_container_width=True)
                    if ask_btn:
                        clean_q = question_text.strip()
                        if not clean_q:
                            st.warning("Please enter a question.")
                        else:
                            st.session_state.current_question_text = clean_q
                            st.session_state.last_error_retrieved = None
                            st.session_state.last_llm_error = None
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
                        st.warning("This video doesn't seem to answer that question.")
                    else:
                        st.markdown("### Answer")
                        st.markdown(qa_res["answer"])

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
                                    st.markdown(
                                        f"**[{src['n']}]** *({fmt_time(src['start'])}-{fmt_time(src['end'])})* {src['text']}"
                                    )

            # -------------------- TAB 2: TRANSCRIPT --------------------
            with tab_transcript:
                df_records = [
                    {
                        "time (mm:ss)": fmt_time(seg["start"]),
                        "start (s)": seg["start"],
                        "text": seg["text"],
                    }
                    for seg in segments
                ]
                df = pd.DataFrame(df_records)
                st.dataframe(df, use_container_width=True, height=450)

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


if __name__ == "__main__":
    main()
