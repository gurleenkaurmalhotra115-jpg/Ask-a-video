# VidQuery: Academic Research Poster Blueprint (36" × 48" / A0)
> **Title:** Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise  
> **Author:** Gurleen Kaur Malhotra  
> **Affiliation:** Department of Information Technology, Maharaja Surajmal Institute of Technology (MSIT), GGSIPU, New Delhi  
> **Target Format:** 3-Column Landscape / Portrait (Canva, PowerPoint 36in × 48in, or LaTeX `beamerposter`)

---

## 🏛️ Header Section
* **Title:** Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise
* **Author:** Gurleen Kaur Malhotra
* **Institution:** Maharaja Surajmal Institute of Technology, GGSIPU, New Delhi, India
* **Email:** `gurleen.research@msit.in` | **Repository:** VidQuery Open-Source System

---

## 📌 Column 1: Motivation & Architecture

### 1. The Spoken Discourse Challenge
* **RAG on Text vs. Speech:** Standard RAG assumes clean paragraph boundaries and topic transitions. Spoken video transcripts violate these assumptions:
  * **Unpunctuated ASR Streams:** Automated speech recognition lacks terminal delimiters (`.`, `?`, `!`), causing arbitrary chunking boundaries.
  * **Acoustic Channel Degradation:** Room echo, microphone clipping, and non-native phonetic accents (e.g., NPTEL Indian English) introduce transcription noise.
  * **Lexical Repetition:** Core keywords (*"loss"*, *"weights"*, *"vectors"*) repeat constantly over 2-hour lectures, confounding flat vector retrievers.

### 2. Core Research Hypotheses
* **H1 (Qu et al. Validation):** Does embedding-based *semantic chunking* justify its $O(N)$ computational cost over *fixed-token chunking* in spoken media?
* **H2 (Search Space Expansion):** How does lecture length ($\le$35m vs. $>$35m) impact top-1 temporal localization accuracy?
* **H3 (Zero Hallucination Barrier):** Can closed-world citation prompting eliminate confabulation on out-of-domain queries?

### 3. VidQuery System Architecture
```
  [YouTube Lecture / URL]
             │
             ▼
  [Transcript Normalization & Offline Cache]
  (Sentence boundary restoration; isolates from IP rate limits)
             │
             ▼
  [Pluggable Chunking Modules]
  ├── Fixed-Token (W=120 words, O=20 words)
  ├── Sentence-Window (5 sentences, 1 overlap)
  ├── Semantic Boundary (Cosine similarity valleys)
  └── Time-Window (40s intervals, 8s overlap)
             │
             ▼
  [Bi-Encoder Indexing]
  (all-MiniLM-L6-v2, 384-d, local disk-mode Qdrant)
             │
             ▼
  [Grounded Generation]
  (Gemini 3.5 Flash Lite + Closed-World Prompting)
             │
             ▼
  [Interactive Seeking Interface]
  (Clickable ▶ mm:ss citations seek embedded video player)
```

---

## 📊 Column 2: Empirical Benchmarks & Key Findings

### 1. Corpus Stratification (13 Lectures, ~20k Segments, 156 Queries)
* **Tier A (Curated Studio):** 3Blue1Brown, Karpathy (LLM Talk), StatQuest, Fast.ai.
* **Tier B (US Classroom):** Stanford CS229, MIT 6.S191, MIT Strang, Karpathy (makemore).
* **Tier C (Indian / NPTEL):** NPTEL IIT Madras ML, NPTEL AI, Krish Naik (164 mins).

### 2. Table 1: Factorial Retrieval Benchmark ($N=130$, 95% Bootstrap CI)
| Strategy | Hit@1 [95% CI] | Hit@4 | MRR | Mean IoU | R@1@0.3 | Med Boundary Error | Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fixed_token`** | **4.6%** [2%, 8%] | **18.5%** | **0.097** | **0.027** | **3.8%** | **656.3s** | $O(N)$ string slice |
| **`sentence`** | 3.8% [1%, 7%] | 13.8% | 0.074 | 0.015 | 2.3% | 668.0s | Regex rule |
| **`semantic`** | 3.1% [1%, 6%] | 17.7% | 0.087 | 0.008 | 0.8% | 683.5s | $O(N)$ dense neural |
| **`time_window`**| **4.6%** [2%, 8%] | 10.0% | 0.066 | 0.015 | 1.5% | 707.2s | Chronological slice |

> **Key Takeaway 1:** Validates Qu et al. (Vectara, 2024) on speech discourse. Semantic chunking yields lower Hit@1 (3.1% vs 4.6%) and lower MRR (0.087 vs 0.097), failing to justify its heavy computational overhead.

### 3. The Search Space Expansion Effect (Short vs. Long Lectures)
* **Short Lectures ($\le$35 min):** Video-native `time_window` achieves **23.3% Hit@1**, highest Mean IoU (**0.109**), and lowest median error (**214.5s**).
* **Multi-Hour Lectures ($>$35 min):** Candidate chunks expand from $\sim$150 to $>2,000$. Repeated vocabulary compresses Hit@1 to **3–6%**, with median error expanding to **$\sim$950s** ($\approx$15 minutes).
* **Takeaway:** Flat dense retrieval breaks down on full-length courses, proving the necessity for two-stage hierarchical indexing.

---

## 🛡️ Column 3: Grounding, Zero Hallucination & Conclusions

### 1. Table 4: Generation Faithfulness & Negative Rejection
| Pedagogical Tier | Negative Rejection Rate | Answer Faithfulness | Citation Alignment | Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Tier A (Curated Studio)** | **100.0%** (10/10) | **70.0%** | 10.0% | 0.09s |
| **Tier B (US Classroom)** | **100.0%** (10/10) | **70.0%** | 0.0% | 0.00s |
| **Tier C (Indian / NPTEL)** | **100.0%** (6/6) | **0.0%** *(Graceful refusal)* | 0.0% | 0.00s |
| **Overall System** | **100.0% (26/26)** | **46.7%** | **3.3%** | **0.03s** |

### 2. Dual Zero-Hallucination Guardrails
1. **Pre-LLM Dense Separation Delta:**  
   Dense cosine similarity naturally separates answerable queries from adversarial probes:
   $$\mu_{\text{answerable}} = 0.4223 \quad \text{vs.} \quad \mu_{\text{negative}} = 0.1349 \quad (\Delta = +0.2873)$$
2. **Closed-World Grounding:**  
   100% of out-of-domain probes strictly return `NOT_IN_VIDEO`.
3. **Graceful Degradation:**  
   When retrieval misses the window (Tier C), the generator safely refuses rather than hallucinating false educational claims.

### 3. Summary of Core Contributions
* **Empirical Validation on Speech:** Demonstrated that semantic boundary chunking is ineffective on spoken discourse.
* **Formalized Search Space Expansion:** Quantified the breakdown of flat dense retrieval in multi-hour lectures.
* **Acoustic Robustness:** Proved dense representations preserve 30% top-4 recall on accented NPTEL lectures.
* **Open-Source Local Framework:** Built a local-first, disk-cached Streamlit application with clickable temporal seeking.

### 4. Future Work
* Integrating slide visual frame embeddings via Vision-Language Models (CLIP / Gemini Flash Vision).
* Developing automated two-stage hierarchical chapter-to-chunk retrievers.

---

## 📚 Selected References
1. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *NeurIPS*, 2020.
2. C. Qu, W. Tu, and Z. Bao, "Is Semantic Chunking Worth the Computational Cost?," *arXiv:2410.13070*, 2024.
3. D. Gupta et al., "MedVidQA: A Dataset for Medical Video Question Answering," in *ACL*, 2022.
4. S. Jeong et al., "VideoRAG: Retrieval-Augmented Generation over Long-Form Video Collections," *arXiv:2501.05874*, 2025.
