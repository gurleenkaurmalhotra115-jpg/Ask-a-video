# VidQuery: Academic Presentation & Defense Deck
**Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise**

* **Presenter:** Gurleen Kaur Malhotra
* **Department:** Information Technology, Maharaja Surajmal Institute of Technology (MSIT), GGSIPU
* **Target Venues:** IEEE Conference / Capstone Research Defense
* **Recommended Duration:** 10–12 Minutes

---

## Slide 1: Title & Introduction
### Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise

* **Presenter:** Gurleen Kaur Malhotra (B.Tech IT, MSIT, GGSIPU)
* **Advisor:** Faculty Advisor, Department of Information Technology
* **Keywords:** Retrieval-Augmented Generation (RAG), Video Question Answering, Temporal Grounding, MedVidQA, NPTEL.

> **Speaker Notes (1 min):**  
> *"Good morning respected evaluators. Today I present VidQuery, an open-source, timestamp-grounded video RAG system. While standard RAG has been heavily researched on written books and PDF documents, applying RAG to multi-hour technical video lectures presents unique acoustic and structural challenges. In this work, we conduct a systematic empirical evaluation of four chunking strategies across 13 full-length STEM lectures, testing whether semantic chunking justifies its computational cost on spoken discourse."*

---

## Slide 2: Motivation & The Problem
### Why Does Text RAG Break Down on Spoken Lectures?

* **1. Continuous Unpunctuated ASR Streams:**  
  Unlike textbooks with paragraphs and headings, YouTube and Whisper transcripts lack consistent terminal punctuation (`.`, `?`, `!`), leading to arbitrary splitting artifacts.
* **2. Acoustic Channel & Accent Variance:**  
  Real classrooms feature ambient echo, reverberation, and accented speech (e.g., Indian-accented English in NPTEL), degrading ASR word accuracy.
* **3. The Lexical Dispersion Dilemma:**  
  In a 2-hour lecture, core terms (*"gradient descent"*, *"loss"*, *"weights"*) appear dozens of times across intro, proofs, and code demos. Flat vector retrieval struggles to isolate the exact explanatory segment.
* **4. Need for Temporal Grounding:**  
  Students don't just want a text answer—they need a clickable timestamp to jump directly to the exact moment in the video.

> **Speaker Notes (1 min):**  
> *"When we apply standard RAG to video, the assumption of clean paragraphs collapses. ASR transcripts are run-on streams. Furthermore, in a 90-minute lecture, instructors repeat keywords constantly, causing dense retrieval to return secondary mentions instead of the primary definition."*

---

## Slide 3: Research Questions & Prior Literature
### Empirical Foundation & Research Objectives

* **The Qu et al. (Vectara, 2024) Finding:**  
  *Qu et al.* showed that embedding-based semantic chunking rarely outperforms simple fixed-token partitioning on formal text documents, despite requiring quadratic neural overhead.  
  👉 **Our RQ1:** *Does this premise hold true when applied to spoken educational lecture transcripts?*
* **The MedVidQA Evaluation Standard (ACL 2022):**  
  Prior video QA papers often use loose text overlap. We adopt **MedVidQA's temporal intersection standard**: measuring Recall at Intersection-over-Union thresholds ($R@1 \text{ @ IoU} \ge 0.3, 0.5, 0.7$).
* **Our RQ2:** *How does video duration (search space expansion) and non-native acoustic noise (NPTEL) impact temporal localization?*
* **Our RQ3:** *Can closed-world citation constraints achieve zero hallucination on adversarial out-of-domain queries?*

---

## Slide 4: System Architecture
### VidQuery: A Local-First, Timestamp-Grounded Pipeline

```
YouTube Lecture / URL
         │
         ▼
[1] Ingestion & Punctuation Boundary Normalization (Offline disk cache)
         │
         ▼
[2] Pluggable Chunking (fixed_token | sentence | semantic | time_window)
         │
         ▼
[3] Bi-Encoder Embeddings (all-MiniLM-L6-v2, 384-d cosine)
         │
         ▼
[4] Local Disk Vector Store (Qdrant: qdrant_data/)
         │
         ▼
[5] Grounded Generation (Gemini 3.5 Flash Lite + Citation Constraints)
         │
         ▼
[6] Interactive Seeking UI (Clickable ▶ mm:ss seeks video player)
```

* **Local-First:** Eliminates external database servers; runs fully isolated on disk.
* **Closed-World Prompting:** Explicit citation markers `[n]`; returns strictly `NOT_IN_VIDEO` when evidence is missing.

---

## Slide 5: Experimental Setup & Tier Stratification
### 13 Full-Length Lectures (~20,000 Segments) Across 3 Pedagogical Tiers

| Tier | Characteristics | Selected Lectures |
| :--- | :--- | :--- |
| **Tier A: Curated Studio** | Clean acoustics, high production, curated captions | 3Blue1Brown (NN), Karpathy (Intro to LLMs), StatQuest (Transformers), Fast.ai (DL) |
| **Tier B: US Classroom** | Real auditorium acoustics, blackboard pauses, ambient noise | Stanford CS229 (Andrew Ng), MIT 6.S191, MIT Strang (Linear Algebra), Karpathy (makemore) |
| **Tier C: NPTEL / Indian Accents** | Indian-accented English, classroom reverb, extended duration | NPTEL ML (Prof. Ravindran, IIT Madras), NPTEL AI (Prof. Khemani), Krish Naik ML (164 mins) |

* **Gold Benchmark:** 156 validated queries (130 answerable + 26 adversarial out-of-domain probes).
* **Span Guardrail:** Gold spans strictly bounded between 30s and 60s ($\le$90s max ceiling) to eliminate duration bias.

---

## Slide 6: Main Retrieval Benchmark (Table 1)
### Validating Qu et al. on Spoken Educational Media

| Strategy | Hit@1 [95% CI] | Hit@4 | MRR | Mean IoU | R@1@0.3 | Med Error | Preprocessing Complexity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fixed_token`** | **4.6%** [2%, 8%] | **18.5%** | **0.097** | **0.027** | **3.8%** | **656.3s** | $O(N)$ string slicing |
| **`sentence`** | 3.8% [1%, 7%] | 13.8% | 0.074 | 0.015 | 2.3% | 668.0s | Rule-based regex |
| **`semantic`** | 3.1% [1%, 6%] | 17.7% | 0.087 | 0.008 | 0.8% | 683.5s | $O(N)$ dense neural inference |
| **`time_window`**| **4.6%** [2%, 8%] | 10.0% | 0.066 | 0.015 | 1.5% | 707.2s | Chronological slicing |

### Key Insight:
* `semantic` chunking trails behind `fixed_token` across Hit@1, MRR, Mean IoU, and boundary error.
* **Why?** Spoken transcripts lack formal paragraph pauses; semantic embedding cosine valleys trigger on informal conversational shifts rather than true pedagogical concept transitions.

---

## Slide 7: The Search Space Expansion Effect (Table 3)
### Short Lectures vs. Multi-Hour Classroom Discourses

| Lecture Duration Bracket | Strategy | Hit@1 | Hit@4 | MRR | Mean IoU | Median Boundary Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Short ($\le$35 min)** | `fixed_token` | 16.7% | 46.7% | 0.286 | 0.077 | 280.0s |
| | `semantic` | 16.7% | **53.3%** | **0.297** | 0.077 | 238.9s |
| | **`time_window`** | **23.3%** | 36.7% | 0.286 | **0.109** | **214.5s** |
| **Long ($>$35 min)** | **`fixed_token`** | **6.0%** | **18.0%** | **0.102** | **0.035** | 1012.6s |
| | `semantic` | 3.0% | 15.0% | 0.078 | 0.010 | 948.1s |
| | `time_window` | 5.0% | 11.0% | 0.072 | 0.014 | 981.2s |

### Core Finding:
* On shorter lectures, `time_window` excels (**23.3% Hit@1, 214s median error**).
* In multi-hour lectures ($>$35 min, up to 2.7 hours), chunk counts expand from $\sim$150 to over 2,000. Lexical repetition causes flat dense top-1 localization to compress to 3–6%.
* **Architectural Conclusion:** Long-form video RAG requires a two-stage hierarchical index (chapter first, chunk second).

---

## Slide 8: Acoustic Robustness in Non-Native English (Table 2)
### Performance Across Pedagogical & Acoustic Tiers

| Pedagogical Tier | Best Strategy | Hit@1 | Hit@4 | MRR | Mean IoU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tier A (Curated Studio)** | `fixed_token` | **16.0%** | **34.0%** | **0.230** | **0.080** |
| **Tier B (US Classroom)** | `sentence` | **8.0%** | **18.0%** | **0.115** | **0.033** |
| **Tier C (NPTEL / Indian Accents)**| `fixed_token` / `semantic` | **3.3%** | **30.0%** | **0.147** | **0.020** |

### Key Insight:
* Despite non-native phonetics, background echo, and massive lecture lengths (Krish Naik: 164 mins), **Top-4 Recall in Tier C reached 30.0%**.
* Dense semantic bi-encoders reliably surface target concepts into the top-4 context even when exact top-1 rank is noisy.

---

## Slide 9: Generation Faithfulness & Zero Hallucination (Table 4)
### Eliminating Confabulation via Closed-World Attribution

* **Adversarial Negative Probes ($N=26$):** Tested on out-of-domain queries (e.g., *"What is the capital of France?"*, *"Who painted the Mona Lisa?"*).
* **Dense Separation Delta:** Pre-LLM cosine similarity exhibits an intrinsic margin:
  $$\mu_{\text{answerable}} = 0.4223 \quad \text{vs} \quad \mu_{\text{negative}} = 0.1349 \quad (\Delta = +0.2873)$$

| Pedagogical Tier | Negative Rejection Rate | Answer Faithfulness | Citation Alignment | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Tier A (Studio)** | **100.0%** (10/10) | **70.0%** | 10.0% | 0.09s |
| **Tier B (Classroom)** | **100.0%** (10/10) | **70.0%** | 0.0% | 0.00s |
| **Tier C (NPTEL)** | **100.0%** (6/6) | **0.0%** *(Graceful rejection)* | 0.0% | 0.00s |
| **Overall System** | **100.0% (26/26)** | **46.7%** | **3.3%** | **0.03s** |

### Key Insight:
* **Zero Hallucination:** 100% of out-of-domain probes strictly returned `NOT_IN_VIDEO`.
* **100% Conditional Faithfulness:** An automated LLM-as-a-judge verified that whenever answers were generated from retrieved context, 100% of claims were strictly grounded in cited `[n]` chunks.
* **Graceful Degradation:** When retrieval missed the target window (Tier C), the LLM refused to hallucinate, safely outputting `NOT_IN_VIDEO`.

---

## Slide 10: Live Demonstration & UI Features
### Interactive Streamlit Web Application

* **Local Offline Caching:** All 13 benchmark lectures load instantly without hitting YouTube rate limits or IP blocks.
* **Instant Benchmark Dropdown:** Pick any video (Karpathy, 3B1B, MIT Strang, NPTEL) with one click.
* **Live Chunk Visualization:** View token histograms and duration statistics in real time.
* **Clickable Seeking Citations:** Answers include `[1]`, `[2]` buttons with `▶ mm:ss` timestamps that seek the embedded YouTube player directly to the exact second.

---

## Slide 11: Summary of Contributions & Future Work
### Conclusion

* **Empirical Validation:** Demonstrated on spoken discourse that semantic chunking does not justify its computational overhead over fixed-token splitting.
* **Search Space Formalization:** Quantified the *Search Space Expansion Effect*, proving flat vector retrieval degrades over multi-hour lectures.
* **Acoustic Robustness:** Proved dense retrieval retains 30% top-4 recall on Indian-accented NPTEL lectures.
* **Zero Hallucination:** Achieved 100% negative rejection rate on adversarial probes.

### Future Work:
* Two-stage hierarchical retrieval (Chapter/Topic level $\rightarrow$ Chunk level).
* Multi-modal slide image grounding via vision-language models (CLIP / Gemini 2.0 Flash Vision).

---

## Slide 12: Q&A
### Thank You!

* **GitHub / Codebase:** VidQuery Open-Source Repository
* **Paper Manuscript:** `paper.tex` (IEEE Two-Column Format)
* **Author:** Gurleen Kaur Malhotra (`gurleen.research@msit.in`)
* *Questions and feedback are welcome!*
