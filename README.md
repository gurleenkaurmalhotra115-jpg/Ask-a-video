# VidQuery ("Ask a Video")

[![VidQuery CI](https://github.com/gurleenkaurmalhotra115-jpg/Ask-a-video/actions/workflows/ci.yml/badge.svg)](https://github.com/gurleenkaurmalhotra115-jpg/Ask-a-video/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Vector DB: Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-DC244C)
![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

> **Timestamp-Grounded Video RAG & Empirical Chunking Evaluation**  
> *Author:* Gurleen Kaur Malhotra, B.Tech IT, Maharaja Surajmal Institute of Technology (MSIT), GGSIPU, New Delhi.  
> 📖 **[Read the Full Technical Wiki & Viva Defense Guide (WIKI.md)](WIKI.md)** | 📄 **[Read the IEEE Research Paper (paper.pdf)](paper.pdf)**

---

## 📌 About

**VidQuery ("Ask a Video")** is an open-source, local-first Retrieval-Augmented Generation (RAG) framework for long-form educational STEM lectures. Given a YouTube video URL or pre-indexed lecture, VidQuery:
1. Ingests timestamped transcripts (via YouTube captions or local Whisper ASR fallback).
2. Normalizes punctuation boundaries to avoid ASR splitting artifacts.
3. Partitions transcripts using four pluggable chunking strategies (`fixed_token`, `sentence`, `semantic`, and `time_window`).
4. Indexes dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) in local disk-mode Qdrant.
5. Guards generation with a configurable **Retrieval-Confidence Gate** (`MIN_RETRIEVAL_CONFIDENCE`) to intercept out-of-domain or weak-match queries *before* calling the LLM.
6. Generates citation-constrained answers via Google Gemini with an **ordered multi-model fallback chain** (`GEMINI_FALLBACK_MODELS`), retries, and disk caching.
7. Surfaces interactive `▶ mm:ss` buttons that dynamically seek an embedded video player to the exact referenced second.

The repository also includes an empirical research study formatted for an **IEEE conference paper** (`paper.tex`), evaluating retrieval quality under MedVidQA temporal intersection metrics ($R@1 \text{ @ IoU} \ge 0.3/0.5/0.7$) across 13 full-length lectures and 156 queries.

---

## 🔬 Key Research Findings

1. **Validation of Qu et al. (Vectara, 2024) on Spoken Discourse**:
   - Across 13 full-length STEM lectures (~20,000 segments) and 156 validated questions, heavy `semantic` chunking failed to outperform simple `fixed_token` partitioning (Hit@1: **4.6% vs 3.1%**, MRR: **0.097 vs 0.087**), while incurring $O(N)$ transformer overhead.
2. **Search Space Expansion Effect**:
   - On short lectures ($\le$35 min), video-native `time_window` partitioning achieves **23.3% Hit@1** and the lowest median boundary error (214.5s).
   - On multi-hour classroom discourses ($>$35 min), lexical dispersion causes flat dense top-1 localization to drop to 3–6%, proving the need for hierarchical chapter-to-chunk indexation.
3. **Acoustic Robustness (NPTEL / Indian Accents)**:
   - Despite accented speech and reverberant classroom acoustics, top-4 recall remained strong at **30.0%** in Tier C.
4. **Zero-Hallucination & Negative Rejection**:
   - Dense retrieval provides a **$\Delta = +0.2873$ margin** between answerable queries ($\mu = 0.4223$) and negatives ($\mu = 0.1349$).
   - Closed-world prompting delivers **100% Negative Rejection Accuracy** on 26 adversarial out-of-domain probes and **100% conditional faithfulness** on retrieved evidence.

---

## 🏗️ Architecture

```
YouTube URL / ID
       │
       ▼
Transcript Ingestion (Captions API / Whisper ASR fallback)
       │
       ▼
Punctuation & Boundary Normalization (Offline JSON Cache)
       │
       ▼
Pluggable Chunking (fixed_token | sentence | semantic | time_window)
       │
       ▼
Dense Embeddings (all-MiniLM-L6-v2, 384-d cosine)
       │
       ▼
Local Vector Store (Qdrant disk mode: qdrant_data/)
       │
       ▼
Top-k Retrieval ──► Grounded Prompt ──► Gemini 3.5 Flash Lite
                                              │
                                              ▼
                                 Interactive Streamlit UI
                          (Clickable ▶ mm:ss Video Seeking)
```

---

## 📂 Project Structure

```
├── app.py                      # Main Streamlit web application (includes Benchmark tab & PDF downloads)
├── launch_demo.bat             # One-click Windows launcher (starts server & opens browser)
├── presentation.html           # Standalone interactive 12-slide HTML presentation deck
├── paper.pdf                   # Pre-compiled camera-ready 4-page IEEE conference PDF
├── paper.tex                   # Publication-ready IEEE 2-column LaTeX manuscript
├── references.bib              # Formal BibTeX bibliography
├── FUTURE_WORK.md              # Hierarchical indexing & multimodal research roadmap
├── multimodal.py               # Visual keyframe extraction & thumbnail synchronization
├── chunking.py                 # 4 pluggable chunking algorithms & stats
├── transcript.py              # Transcript fetcher with offline cache & Whisper
├── embeddings.py               # Bi-encoder embedding generator (all-MiniLM-L6-v2)
├── vectorstore.py              # Local Qdrant vector database operations
├── llm.py                      # Grounded QA with confidence gate, fallback chain & cache
├── eval/
│   ├── corpus_meta.json        # 13-video corpus metadata across 3 tiers
│   ├── questions.csv           # 156 validated questions (130 pos + 26 neg)
│   ├── common.py               # Dataset loaders and validation schemas
│   ├── metrics.py              # MedVidQA temporal IoU & recall metrics
│   ├── evaluate.py             # 4x2 Factorial benchmark & bootstrap CIs
│   ├── eval_generation.py      # LLM judge faithfulness & negative rejection
│   └── ingest_corpus.py        # Automated corpus freezing tool
├── data/transcripts/captions/  # 13 frozen transcript JSON files
├── results/                    # Benchmark CSVs and generated IEEE LaTeX tables
│   ├── latex_table_main.tex    # Table 1: Factorial retrieval benchmark
│   ├── latex_table_tiers.tex   # Table 2 & 3: Tier and duration breakdown
│   ├── latex_table_generation.tex # Table 4: Generation faithfulness
│   ├── retrieval_summary.csv   # Aggregated retrieval metrics
│   └── generation_eval_summary.csv # Generation evaluation metrics
├── tests/                      # Full pytest test suite (53 passing tests)
├── api.py                      # FastAPI REST backend (/health, /videos/*, /benchmarks)
├── docker-compose.yml          # Full stack (FastAPI, Streamlit, Postgres, Redis)
├── .github/workflows/ci.yml    # GitHub Actions CI (verify_all.py + pytest)
├── Dockerfile                  # Production container image (Python 3.11 + ffmpeg)
├── LICENSE                     # MIT License
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quick Start

### Option A: Local Python Setup
Clone the repository and set up a Python 3.11 virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the project root (see `.env.example`):
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite
MIN_RETRIEVAL_CONFIDENCE=0.22
```

Launch the Streamlit app:
```bash
streamlit run app.py --server.port 8502
```
Open **`http://localhost:8502`** in your browser. Choose any lecture from the pre-indexed benchmark dropdown to test offline without hitting YouTube rate limits!

### Option B: Docker & Full Production Stack (FastAPI + Streamlit + Postgres + Redis)
```bash
docker compose up --build
```
- **Streamlit UI:** `http://localhost:8501`
- **FastAPI Docs (Swagger):** `http://localhost:8000/docs`

Or run the single-container UI image:
```bash
docker build -t vidquery .
docker run -p 8501:8501 --env-file .env vidquery
```

### Option C: Model Context Protocol (MCP) Server
VidQuery ships with a native **Model Context Protocol (MCP)** server, enabling any MCP-compliant AI agent (Claude Desktop, Cursor, Gemini CLI) to search timestamped video moments, query lectures with confidence gating, and verify citations:

```bash
python mcp_server.py
```

**Claude Desktop Configuration (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "vidquery": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/path/to/Ask-a-video",
      "env": {
        "GEMINI_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Exposed MCP Tools:**
1. `search_video_moments(video_id, query, k=4, strategy='fixed_token')`: Semantic search returning timestamped video chunks.
2. `ask_grounded_question(video_id, question, k=4, min_confidence=0.22)`: Confidence-gated Q&A with multi-model fallback and exact citations.
3. `verify_citations(question, answer, chunk_texts)`: Independent verification agent checking factual grounding.

### Run Verification & Unit Tests
```bash
python verify_all.py
pytest tests/ -v
```
*58/58 tests passing.*

---

## 📊 Benchmark Reproduction

To reproduce all tables from the paper:

1. **Retrieval Benchmark (Table 1, 2, 3)**:
   ```bash
   python -m eval.evaluate --k 4 --bootstrap 1000
   ```
2. **Generation Faithfulness & Negative Rejection (Table 4)**:
   ```bash
   python -m eval.eval_generation --strategy fixed_token --k 4 --samples-per-tier 10
   ```
Generated LaTeX tables will be saved directly into `results/`.

---

## 📜 Paper Citation

```bibtex
@inproceedings{malhotra2026vidquery,
  title={Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise},
  author={Malhotra, Gurleen Kaur},
  booktitle={Proceedings of Independent Undergraduate Academic Research, MSIT, GGSIPU},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
