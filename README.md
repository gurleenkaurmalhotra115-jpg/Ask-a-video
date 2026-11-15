# VidQuery ("Ask a Video")
> **Timestamp-Grounded Video RAG & Empirical Chunking Evaluation**  
> *Author:* Gurleen Kaur Malhotra, B.Tech IT, Maharaja Surajmal Institute of Technology (MSIT), GGSIPU, New Delhi.  
> 📖 **[Read the Full Technical Wiki & Viva Defense Guide (WIKI.md)](WIKI.md)** | 📄 **[Read the IEEE Research Paper (paper.pdf)](paper.pdf)**

---

## 📌 Overview

**VidQuery** is an open-source, local-first Retrieval-Augmented Generation (RAG) framework designed for long-form educational STEM lectures. Given a YouTube video URL or pre-indexed lecture, VidQuery:
1. Ingests timestamped transcripts (via YouTube captions or local Whisper ASR fallback).
2. Normalizes punctuation boundaries to avoid ASR splitting artifacts.
3. Partitions transcripts using four pluggable chunking strategies (`fixed_token`, `sentence`, `semantic`, and `time_window`).
4. Indexes dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) in local disk-mode Qdrant.
5. Performs top-$k$ semantic retrieval and generates citation-constrained answers using Gemini 3.5 Flash Lite.
6. Surfaces interactive `▶ mm:ss` buttons that dynamically seek an embedded video player to the exact referenced second.

The project includes an empirical research study formatted for an **IEEE conference paper** (`paper.tex`), evaluating retrieval quality under MedVidQA temporal intersection metrics ($R@1 \text{ @ IoU} \ge 0.3/0.5/0.7$) and testing whether semantic chunking justifies its computational overhead on spoken lecture discourse.

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
├── llm.py                      # Grounded QA with Gemini, retries, and disk cache
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
├── tests/                      # Full pytest test suite (43 passing tests)
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up a Python 3.11 virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODEL=gemini-3.8-flash
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py --server.port 8502
```
Open **`http://localhost:8502`** in your browser. Choose any lecture from the pre-indexed benchmark dropdown to test offline without hitting YouTube rate limits!

### 4. Run Unit Tests
```bash
pytest
```
*43/43 tests passing.*

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
