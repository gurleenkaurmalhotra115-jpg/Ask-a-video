# Ask a Video (VidQuery): timestamp-aware video RAG for learning
Goal: user pastes a YouTube lecture URL, asks questions, gets a grounded
answer with clickable timestamps. Also a research project: comparing
chunking strategies for timestamp retrieval.

Stack: Python, Streamlit, youtube-transcript-api (primary transcript
source), yt-dlp + openai-whisper (fallback only), sentence-transformers
(all-MiniLM-L6-v2), Qdrant (local mode, no server), Gemini API
(google-genai). API key via .env (GEMINI_API_KEY).

Files: app.py, transcript.py, chunking.py, vectorstore.py, llm.py,
evaluate.py (later), README.md.

Design rules:
- Transcript segments format everywhere: {"start": float, "duration": float, "text": str}
- chunking.py exposes 4 pluggable strategies behind one interface:
  fixed_token, sentence, semantic, time_window. Each chunk keeps
  start_time and end_time.
- The UI will have a chunking-strategy selector and a Compare mode that
  shows retrieved segments for all 4 strategies side by side.
- Build ONLY the step I ask for. Keep code clean, typed, with short
  docstrings. Show user-friendly errors, never raw tracebacks in the UI.
- After each step, tell me exactly how to test it.
