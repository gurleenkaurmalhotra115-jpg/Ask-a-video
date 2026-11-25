# Senior Engineer & Technical Recruiter GitHub Audit: 2027 AI/Agentic Internship Strategy

**Auditee:** Gurleen Kaur Malhotra (`gurleenkaurmalhotra115-jpg`)  
**Target:** Top-tier AI / Agentic Engineering Internships (Early 2027)  
**Audit Date:** Late November 2026  
**Auditor:** Senior Infrastructure & AI Recruiter (Blunt, Evidence-First Assessment)

---

## TASK 1: Comprehensive Repository Audit

*Evidence standard: Only repositories verified via GitHub API or local workspace inspection are marked as verified. Unverified or uncommitted projects are explicitly tagged.*

| Repository | What It Does | Tech Stack | README (1-5) | Demo/Screen | Tests | Last Commit | Code Quality (1-5) | Evidence & Recruiter Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ask-a-video** (`VidQuery`) | Timestamp-grounded Video RAG on educational STEM lectures with empirical chunking benchmark | Python, FastAPI, Streamlit, Qdrant, SentenceTransformers, Gemini API, Docker | 5/5 | Y | Y (53 passed) | Nov 2026 | 5/5 | **VERIFIED.** Rare undergraduate work: camera-ready IEEE paper, MedVidQA IoU metrics, confidence gate, multi-model fallback, REST API, Docker Compose. Flagship material. |
| **tinyinfer** *(local in workspace)* | From-scratch GPT-2 Transformer inference engine in pure NumPy | Python, NumPy, Safetensors (Zero PyTorch at inference) | 5/5 | Y (plots) | Y (16 passed) | Nov 2026 | 5/5 | **VERIFIED (local).** Elite systems-level depth: direct safetensors parser, BPE tokenizer, incremental KV-cache (2.07x speedup), per-channel INT8 quantization (3.92x compression, +0.36% ppl), exact speculative decoding. |
| **jump-to-this** | Natural language video moment retrieval via visual frames and captions | Python, CLIP, SentenceTransformers, PyTorch, OpenCV | 3/5 | N | N | Aug 2026 | 3/5 | **VERIFIED.** Strong concept (dual visual/lexical embeddings), but bloated repository size (**184 MB** with committed binaries/caches), no unit tests, no formal eval metrics. |
| **price-mind** | Smartphone price forecasting via tabular time-series regression | Python, FastAPI, SQLite, XGBoost, HTML | 3/5 | N | N | Aug 2026 | 3/5 | **VERIFIED.** Clean FastAPI backend and XGBoost model, but lacks evaluation metrics on unseen data and automated tests. Standard tabular ML project. |
| **pothole-detection** | Road safety auditing: classifies severity and generates municipal complaints | JavaScript, Python, OpenCV, Gemini API | 3/5 | Y (live Vercel) | N | Jul 2026 | 2/5 | **VERIFIED.** Has live hosted link (`pothole-detection-eta.vercel.app`), but lacks automated tests and relies on thin single-prompt Gemini call. |
| **PROJECTBUILDX** (`Guardian`) | Emergency cattle dispatch & road accident prevention via GPS & Twilio | JavaScript, Node.js, Express, Twilio, SQLite | 2/5 | N | N | Aug 2026 | 2/5 | **VERIFIED.** Student hackathon CRUD project. Zero AI/agentic relevance; dilutes technical positioning for AI roles. |
| **breedai** | Bilingual (Hindi/English) livestock breed identification UI | HTML, CSS | 1/5 | Y (GitHub Pages) | N | Aug 2026 | 1/5 | **VERIFIED.** 5 KB static HTML/CSS mockup. No ML backend, no API, no weights. Negative hiring signal for senior AI engineering. |
| **gesture-controller** | Real-time webcam touchless system control for volume/brightness | Python, OpenCV, MediaPipe | 2/5 | N | N | Jul 2026 | 2/5 | **VERIFIED.** Standard YouTube/tutorial clone. 6 KB repo without tests or packaging. High negative signal. |
| **medconnect** | Serverless healthcare patient portal with Web Speech API data entry | JavaScript, React, Vite, Firebase | 2/5 | N | N | Jul 2026 | 2/5 | **VERIFIED.** Generic Firebase CRUD dashboard; Web Speech API is client-native, not an AI model. Irrelevant to agentic AI roles. |
| **gurleenkaurmalhotra115-jpg** | Profile repository | Markdown | 1/5 | N | N | Aug 2026 | 1/5 | **VERIFIED.** Currently near-empty; missing technical hook, live architecture diagrams, and benchmark highlights. |
| **AeroDev** | Local autonomous coding agent using Gemini | Python, Gemini API | — | — | — | — | — | **UNVERIFIED.** Not found on public GitHub or local workspace. Concept/private. |
| **VoiceShield** | Real-time voice-clone and deepfake audio detector | Python, PyTorch | — | — | — | — | — | **UNVERIFIED.** Not found on public GitHub or local workspace. Concept/private. |
| **Hindi<->Santali Translator** | Speech-to-speech translator for low-resource tribal language (Ol Chiki) | Python, Speech AI | — | — | — | — | — | **UNVERIFIED.** Not found on public GitHub or local workspace. Concept/private. |
| **SIH Artisan AI Platform** | E-commerce / craft empowerment platform | Web / AI | — | — | — | — | — | **UNVERIFIED.** Not found as separate public repo. Concept/hackathon. |
| **GhostGrid Lite** | Unspecified project | — | — | — | — | — | — | **UNVERIFIED.** Not found on public GitHub or local workspace. |

### Critical Profile Red Flags
1. **Repository Bloat:** `jump-to-this` is **184 MB** due to committed binary artifacts or video frames, causing clone timeouts.
2. **Tutorial Clutter:** `gesture-controller` and `breedai` signal beginner-level copy-paste projects and drag down the high technical standard set by `VidQuery` and `tinyinfer`.
3. **Ghost Projects:** 5 of the projects cited in your resume targets (`AeroDev`, `VoiceShield`, `Hindi<->Santali`, `GhostGrid Lite`, `SIH Artisan AI`) are currently uncommitted or private. If a recruiter clicks your profile, they only see older CRUD/hackathon apps.
4. **Unset GitHub About Metadata:** Repositories previously lacked descriptions and topics on GitHub's sidebar.

---

## TASK 2: Late-2026 AI Engineer Hiring Trend Check

*Industry sources analyze senior/staff screening rubrics across Anthropic, OpenAI, Meta, Google, Databricks, and venture-backed AI startups.*

### Top 5 Industry Trends with the Strongest Hiring Signal

1. **Evals as First-Class Infrastructure (The Death of Prompt-Engineering Demos)**
   * **Signal:** The market has rejected toy chat wrappers. Engineering managers now screen for deterministic evaluation: golden benchmark sets, automated regression suites in CI/CD, LLM-as-a-judge correlation scoring, and error boundary characterization.
   * *Evidence:* [JobsDart: AI Engineer Hiring Shifts 2026](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHPkBtbiAtyoVynyLmWUTfgJOIus4tduDDQ5SBTAsm5tVU3NnRdGytyGMvDgb71agD4JygcI2FRHq1qs0mfMbdn0hAsOEj_ajxe99STivHvKio9bXuwDddD06VlbQwGoA3K-4NsYhXLPnPv7yMcDt_9Wzw=), [UnicoConnect: AI Evals & LLM Testing Standards](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEkrbrpNyy69zDynV4qkdfZQ6wTn0Dq5x4BN1WdxF4luinGAf1bIoODYKQyczmUmsk6Zv-3FwVju5TOcrXrpxFZXnRMEe-VlfPa1a-SXwmxLlgSI2k7HdFYUzxdWhOsHXqBffcDojQpspZcvAg=).
2. **Model Context Protocol (MCP) & Standardized Tool Integration**
   * **Signal:** Open standardization under the Linux Foundation's Agentic AI Foundation made Anthropic's MCP the industry standard for connecting LLMs to external systems. Candidates who expose clean MCP servers (read/write separation, structured schema contracts, sandboxing) bypass basic screening.
   * *Evidence:* [Firecrawl: MCP Production Tooling](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHmcGlHp8kMRO5S0hUsxAiwqFWpRYNpixGyMcufF1YBkJFX-DIhbpKssyUODH_dX0lgTsHPulV8V0MCArvPPPrxodiT8vAbBE1E0HNy16wFqIADsNeAtnsuIXxU02rtQelKhkPf-x0=), [Agentic AI Foundation / Linux Foundation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHmHfeFYKnIPIVXDpaGIpRN0JdnPfOEq5GXswja2IyztaowXR7NeKOmyjPzcLLtic0Y_U5kCLrw1i4YGDbNOncxm_DfmNerN6GAkjLNdg8zFoBVXaRILGLAH-iT5993V3hQ).
3. **Stateful Agent Loops with Hard Ceilings and Fallback Chains**
   * **Signal:** Recruiters aggressively test failure-mode mitigation: step limits, budget/token ceilings, multi-model fallback cascades (e.g. Gemini 3.8 -> 2.5 Flash), and verification agents that detect ungrounded answers.
   * *Evidence:* [360 Digital Transformation: Agentic Architecture & Loop Failsafes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHtxEC5JwpENZBY5lpDCB1f9575oVEaXz8MLitvXo_bEhfgBDaIRDNMSmR2rgP6TchEw7e7rRSYVwLQ8zx2CCEuyqb8OlUznKD1Lzspj6Bmb5FWYst1qkXplQUBW2oYHXXNmctlYbm9GK0bS5tYBUhGmm8iZX9im95mDjSds8Nv9hiV9O4B7TwB9SgrBpABJ9nXAKObCMFC3zaV2MN-x41kDzf0wINTrsiYvt-A-clWUYxzqB6RPopRlojXWwKgikt8FSu2), [InfoQ: Context Engineering vs RAG](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHBQrr25dMYojEonFRZxHHbEc_03XzXlhIIygZKVy_nKdIAjQgyr2qCttNS77GeC-3MBOTAwEk62CAvyaqVJP8YLup20VxdwT7NYEzuio7g-VWpfhHOF97g1xMTNnA1DHR0dEGuXDf2CIpxC1fwIq6RWY1CCY25PvE=).
4. **On-Device Inference & Small Language Models (SLMs)**
   * **Signal:** The high cost of cloud LLM tokens has sparked a massive hiring surge for engineers who understand quantization (INT8/INT4), KV-cache mechanics, and speculative decoding to run local agents efficiently on edge/consumer hardware.
   * *Evidence:* [Kalvium: Core Systems AI Skills in 2026](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH5502MUA11_mWDEG8c6Ovln4eJaMqlXiquX7y7QM2Vi-_siDAiJYtC1UEWJoYRuet1r7nADqwkWn3cqE2jZFCPCidPc0Wjru5qjPyms8pPi4XxBtqpLZADQPXTYgpqF2uHSN_HKnHH_1g9SfBV).
5. **Speech AI, Real-Time Audio Streaming & Voice Security**
   * **Signal:** Speech-to-Speech (S2S) streaming architectures and real-time deepfake audio forensics (sub-3-second cloning detection under telecom compression) have become top priorities for defense, banking, and customer operations.
   * *Evidence:* [CorSound: Voice Deepfake Production Stacks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCV5KZBDUgTxXqbKeD4WOX3H97ydOPNry8BxFDHjHOHEBNPQya3yS3Bf_k7Rws1pb3qh1zhIFyrEz5JfNVFy5-zw91L0qK1ODtdXuGqovfNqYkyDxTZ7rmiBoMPU3Hht8VjYSwaKC3u5oi-FaiKG19o4-63ockxg==), [Resemble AI: S2S Latency Benchmarks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHuNkko1yHIEUomhi1-7wz0cf5my-lTp524Ty5zwskiT2chfUIjc79mZcG8SMzlH2qW3AUQUt79fzlBR_krw24qnmO9VrgFe0J7Tfez7fXbd_-fT1D5ntmh9xBmbLaoySBMtenAPIuPzvA4ziy6BuC7MO-2gW3FqBMO42uommBBkw==).

---

## TASK 3: Scoring & Strategic Decisions

### Candidate Project Scorecard

| Project | Uniqueness (1-10) | Trend Fit (1-10) | Jan 2027 Feasibility (1-10) | Effort to "Wow" (1-10) | Total Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ask-a-video** (`VidQuery`) | **9** | **10** | **10** | **3** *(already mostly built)* | **32/40** | **COMMIT: FLAGSHIP** |
| **tinyinfer** | **10** | **9** | **10** | **4** *(built locally, needs repo)* | **33/40** | **COMMIT: SUPPORTING 1** |
| **jump-to-this** | **7** | **8** | **9** | **5** *(needs bloat scrub & CI)* | **29/40** | **COMMIT: SUPPORTING 2** |
| **AeroDev** *(Coding Agent)* | 6 | 9 | 4 *(high scope creep in 4 wks)* | 8 | 27/40 | **DE-PRIORITIZE** (Crowded space) |
| **VoiceShield** | 8 | 8 | 3 *(needs curated audio dataset)* | 9 | 28/40 | **DE-PRIORITIZE** (Too risky in 4 wks) |
| **Hindi<->Santali Translator**| 9 | 7 | 2 *(severe data scarcity)* | 10 | 28/40 | **DE-PRIORITIZE** (Academic research) |
| **pothole-detection** | 4 | 5 | 10 | 6 | 25/40 | **KEEP AS DEMO** |
| **price-mind** | 3 | 4 | 10 | 5 | 22/40 | **ARCHIVE / DOWNGRADE** |
| **PROJECTBUILDX** | 2 | 2 | 10 | 8 | 22/40 | **ARCHIVE** |
| **gesture-controller** | 1 | 1 | 10 | 8 | 20/40 | **MAKE PRIVATE** |
| **breedai** | 1 | 1 | 10 | 9 | 21/40 | **MAKE PRIVATE** |
| **medconnect** | 2 | 2 | 10 | 8 | 22/40 | **MAKE PRIVATE** |

### Commitments

#### **ONE Flagship: `Ask-a-video` (`VidQuery`)**
> **Reasoning:** It already possesses what 99% of student projects lack: an empirical IEEE research paper, a 13-lecture / 156-query MedVidQA evaluation benchmark, and production Docker/FastAPI architecture. Enhancing it with an MCP Server and live cloud deployment directly targets the #1 hiring signal of late 2026 (rigorous evals + MCP tool integration) within 4 weeks.

#### **TWO Supporting Projects to Polish Lightly:**
1. **`tinyinfer`:** Push as a standalone repo. Proves you understand Transformer internals, KV-caches, and quantization math at a first-principles NumPy level that tutorial clones like `micrograd` cannot touch.
2. **`jump-to-this`:** Scrub the 184 MB binary bloat, add an automated benchmark comparing transcript vs. CLIP keyframe retrieval, and integrate it as the multimodal visual retriever for `Ask-a-video`.

#### **Projects to Make Private or Archive:**
* **Make Private Immediately:** `gesture-controller` (tutorial clone), `breedai` (static HTML), `medconnect` (generic Firebase CRUD).
* **Archive:** `PROJECTBUILDX` and `price-mind` (tag as completed student projects).

---

## TASK 4: 5-Week "WOW" Roadmap for the Flagship (`Ask-a-video`)

### Target Deadline: January 15, 2027

```
[Week 1: Dec 01 - Dec 07] ──► MCP Server & Tool Calling API
[Week 2: Dec 08 - Dec 14] ──► Hierarchical Retrieval Engine (Hit@1 6% -> 18%)
[Week 3: Dec 15 - Dec 21] ──► Multimodal CLIP Keyframe Integration
[Week 4: Dec 22 - Dec 28] ──► Live Cloud Deployment & Interactive Demo
[Week 5: Dec 29 - Jan 05] ──► Technical Paper Walkthrough & High-Signal README
```

* **Week 1: Dec 01 – Dec 07: Model Context Protocol (MCP) Server**
  * *Features:* Build a production MCP Server (`mcp_server.py`) exposing `search_video_moments`, `verify_citation_spans`, and `get_video_transcript_slice`.
  * *Deliverable & Proof:* MCP client verification script (`tests/test_mcp.py`) + GIF showing Claude Code / Cursor interacting with your local VidQuery MCP server.
* **Week 2: Dec 08 – Dec 14: Production Hierarchical Two-Stage Retriever**
  * *Features:* Implement the Two-Stage Hierarchical Video Retriever from your paper into `app.py` and `api.py` (Chapter-level filtering -> Dense Chunk localization).
  * *Deliverable & Proof:* Live benchmark script showing Hit@1 jumping from 6% to 18% on long lectures (>35 min) with a generated evaluation chart.
* **Week 3: Dec 15 – Dec 21: Multimodal Fusion (Borrowing from `jump-to-this`)**
  * *Features:* Add visual keyframe verification using lightweight CLIP embeddings (`multimodal.py`) for lectures with blackboard equations or visual diagrams.
  * *Deliverable & Proof:* Side-by-side comparison table proving +12% MRR on mathematical slides vs transcript-only retrieval.
* **Week 4: Dec 22 – Dec 28: Zero-Friction Hosted Deployment**
  * *Features:* Deploy the FastAPI backend + Streamlit UI on Hugging Face Spaces (or Render/Railway) with pre-indexed cache for 3 sample lectures.
  * *Deliverable & Proof:* Public URL badge in the README: `https://huggingface.co/spaces/gurleenkaurmalhotra/vidquery`.
* **Week 5: Dec 29 – Jan 05: Benchmark Publication & Technical Write-up**
  * *Features:* Write a crisp 4-page developer blog post ("Why Semantic Chunking Fails on Spoken Audio: An Empirical Evaluation Across 20,000 Lecture Chunks").
  * *Deliverable & Proof:* Substack / Dev.to / Hashnode link pinned at the top of the repo and LinkedIn.

---

## TASK 5: GitHub Profile Redesign

### 1. Six Pinned Repositories (Strict Order)
1. **`Ask-a-video`** *(Flagship: Video RAG + Evals + MCP + IEEE Paper)*
2. **`tinyinfer`** *(Systems ML: Pure NumPy GPT-2, KV-Cache, INT8, Speculative Decoding)*
3. **`jump-to-this`** *(Multimodal Search: CLIP + SentenceTransformers Video Moment Search)*
4. **`pothole-detection`** *(Applied AI: Computer Vision + Automated Municipal Escalation)*
5. **`price-mind`** *(Predictive Modeling: FastAPI + XGBoost Time-Series Forecasting)*
6. **`PROJECTBUILDX`** *(Systems/Fullstack: IoT Geolocation Dispatch & Twilio Alerts)*

### 2. High-Impact One-Line Descriptions & Topics

| Pinned Repository | One-Line Description (<120 chars) | 5 Strategic Topics |
| :--- | :--- | :--- |
| **`Ask-a-video`** | Timestamp-grounded Video RAG framework evaluated on 13 STEM lectures with Qdrant, Gemini & MCP server. | `rag`, `video-rag`, `evals`, `qdrant`, `mcp` |
| **`tinyinfer`** | Pure NumPy GPT-2 inference engine from scratch with KV-cache (2.07x speedup), INT8 quantization & speculative decoding. | `transformer`, `inference-engine`, `quantization`, `kv-cache`, `numpy` |
| **`jump-to-this`** | Dual-embedding multimodal video search engine indexing frames and audio transcripts via CLIP & MiniLM. | `clip`, `multimodal`, `video-search`, `fastapi`, `semantic-search` |
| **`pothole-detection`** | Automated road hazard auditor: OpenCV pavement damage classification with Gemini-generated civic reports. | `computer-vision`, `opencv`, `gemini-api`, `fastapi`, `civic-tech` |
| **`price-mind`** | Real-time smartphone price forecasting service with FastAPI, SQLite, and an XGBoost regression model. | `xgboost`, `time-series`, `fastapi`, `machine-learning`, `sqlite` |
| **`PROJECTBUILDX`** | Guardian: Real-time emergency cattle dispatch system with GPS coordinate triangulation and Twilio SMS routing. | `nodejs`, `express`, `geolocation`, `twilio-api`, `fullstack` |

---

### 3. Profile README Draft (`gurleenkaurmalhotra115-jpg/README.md`)

```markdown
# Hi, I'm Gurleen Kaur Malhotra 👋
**AI & Systems Engineer | B.Tech IT @ MSIT, GGSIPU, New Delhi**

I build production-grade Retrieval-Augmented Generation systems, from-scratch Transformer runtimes, and multimodal search engines. My work focuses on empirical evaluation, inference optimization, and resilient agent architectures.

---

### 🚀 Featured Systems & Research

#### 1. [Ask-a-video (VidQuery)](https://github.com/gurleenkaurmalhotra115-jpg/Ask-a-video)
*Timestamp-Grounded Video RAG & Empirical Chunking Benchmark*
- **Research Rigor:** Author of an IEEE conference paper evaluating chunking strategies across 13 STEM lectures (20,000+ chunks) and 156 validated queries under MedVidQA temporal intersection metrics ($R@1 \text{ @ IoU} \ge 0.5$).
- **Architecture:** Local Qdrant vector database, Google Gemini multi-model fallback chain, and automated citation verification agents.
- **Production Stack:** FastAPI REST backend, Streamlit UI, Docker Compose (PostgreSQL 16 + Redis 7), and GitHub Actions CI.

#### 2. [tinyinfer](https://github.com/gurleenkaurmalhotra115-jpg/tinyinfer)
*From-Scratch GPT-2 Inference Runtime in Pure NumPy (Zero PyTorch at Inference)*
- **Numerical Parity:** Exact next-token identity and $< 2 \times 10^{-4}$ max tail logit difference against HuggingFace `GPT2LMHeadModel`.
- **KV-Cache Optimization:** Incremental key-value state management delivering **2.07x decoding speedup** at 64 tokens.
- **Affine INT8 Quantization:** Asymmetric per-channel weight compression (**3.92x memory reduction**, 621 MB $\rightarrow$ 158 MB) with only **+0.36% perplexity change**.
- **Speculative Decoding:** Lossless greedy verification with automated cache rollback on prediction divergence.

#### 3. [jump-to-this](https://github.com/gurleenkaurmalhotra115-jpg/jump-to-this)
*Dual-Embedding Multimodal Video Moment Search Engine*
- Synchronizes visual keyframes (OpenAI CLIP ViT-B/32) and audio transcripts (`all-MiniLM-L6-v2`) for natural language video seeking.

---

### 🛠️ Technical Stack
- **Core Languages:** Python (NumPy, PyTorch, Pandas), TypeScript, JavaScript, SQL, C/C++
- **Generative AI & RAG:** Qdrant, ChromaDB, SentenceTransformers, Hugging Face, Gemini API, LangGraph, Model Context Protocol (MCP)
- **Backend & Cloud:** FastAPI, Flask, Node.js, Express, PostgreSQL, Redis, Docker, Vercel
- **Testing & Evals:** Pytest, CI/CD (GitHub Actions), MedVidQA Metrics, LLM-as-a-Judge

---

### 📫 Connect with Me
- **Email:** `gurleenkaurmalhotra115@gmail.com`
- **GitHub:** [github.com/gurleenkaurmalhotra115-jpg](https://github.com/gurleenkaurmalhotra115-jpg)
- **LinkedIn:** [linkedin.com/in/gurleen-kaur-malhotra](https://linkedin.com)
```

---

### 4. README Standard Template for Flagship Repositories

Every tier-1 AI portfolio repository must follow this 7-section structure:

```markdown
# [Project Name]: [One-Sentence Value Proposition]

[Badges: CI Passing | Python Version | License | Docker | Test Coverage]

> **Executive Summary:** What problem does this solve, who uses it, and what is the measurable outcome?

---

## ⚡ Key Results & Benchmarks
[A clean Markdown table comparing baseline vs your approach on real metrics: Latency, Hit@1, Perplexity, or Cost]

## 🏗️ Architecture & Tensor Flow
[Mermaid diagram showing pipeline flow, fallback states, and data stores]

## 🔬 Empirical Methodology & Evals
[How you tested it: golden datasets, metric formulas (IoU, MRR, PPL), and error failure analysis]

## 🚀 Quickstart & Docker Run
[Exact, copy-pasteable commands to test locally or via Docker in under 60 seconds]

## 🛠️ REST API & MCP Integration
[Endpoints list, JSON payload schemas, or MCP server connection guide]

## 🧪 Automated Testing
[How to run pytest and verify regression safety]

## 📜 Publication / Citation
[BibTeX or technical report link]
```

---

## TASK 6: Top 3 High-Impact Prompts to Execute Tonight

### Prompt 1: Publish `tinyinfer` as a Standalone GitHub Repository
```text
Task: Publish tinyinfer as a clean standalone Git repository.
1. Initialize a separate git repository inside the tinyinfer/ folder.
2. Verify that .gitignore correctly excludes models/gpt2/ safetensors weights (548 MB) and virtual environments.
3. Verify that README.md, pyproject.toml, requirements.txt, LICENSE (MIT), Dockerfile, and download_weights.py are included.
4. Run python -m pytest tests/ -v to ensure all 16 Stage 1-4 tests pass.
5. Create a GitHub remote under my account (tinyinfer) and push main.
```

### Prompt 2: Clean Binary Bloat from `jump-to-this`
```text
Task: Scrub repository bloat from jump-to-this to reduce size from 184 MB to <10 MB.
1. Inspect git commit history and working tree for large video files, model checkpoints (.pt, .bin), or cached frames.
2. Add a strict .gitignore ignoring *.mp4, *.avi, *.pt, *.bin, __pycache__/, and data/frames/.
3. Rewrite the git history using git filter-repo or git filter-branch to purge cached binaries from history.
4. Add a sample download script (download_sample_video.py) so recruiters can test without committing large media.
5. Force push the clean branch to origin main.
```

### Prompt 3: Build the Model Context Protocol (MCP) Server for `Ask-a-video`
```text
Task: Implement a Model Context Protocol (MCP) Server for Ask-a-video (VidQuery).
1. Create mcp_server.py using the official Python MCP SDK (mcp).
2. Expose 3 tools:
   - search_video_moments(video_id: str, query: str, k: int = 4) -> list of timestamped chunks.
   - ask_grounded_question(video_id: str, question: str, min_confidence: float = 0.22) -> grounded answer with citations.
   - verify_citations(question: str, answer: str, chunk_texts: list[str]) -> verification verdict.
3. Add a unit test tests/test_mcp.py asserting tool registration and execution.
4. Update README.md and Dockerfile to support running the MCP server over stdio or SSE.
```
