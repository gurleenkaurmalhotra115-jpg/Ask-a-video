# Ask-a-Video: Live Demo & Viva Defense Master Cheat-Sheet

> **Examiner / Viva Quick-Reference Guide**  
> Keep this document handy during project defense, capstone review, or conference presentations.

---

## ⚡ 1. The 30-Second Elevator Pitch
> *"Ask-a-Video is an open-source, local-first Video RAG framework that turns multi-hour STEM lectures into interactive, timestamp-grounded knowledge bases. We conducted an empirical factorial evaluation across four chunking strategies on 13 STEM lectures stratified into Curated Studio, Academic Classroom, and Regional Accented tiers. Our core finding confirms Qu et al. on spoken discourse: complex semantic boundary chunking fails to outperform simple fixed-token partitioning ($p = 0.508$, McNemar test) while incurring a 160× preprocessing latency penalty. To address the severe 'Search Space Expansion Effect' in long lectures, we formulated a Two-Stage Hierarchical Video Retriever that recovers Hit@1 from 6% to 18%."*

---

## 🎬 2. The 3-Minute Live Demo Sequence

### **Step 1: Offline Ingestion & Zero-Latency Caching**
* **Select from dropdown:** `Andrej Karpathy - Intro to Large Language Models (zjkBMFhNj_g)`
* **Click:** `Fetch transcript`
* **What to say to Examiners:**  
  *"We built a local-first ingestion engine with disk caching that protects the system from YouTube API quota limits and network jitter. Over 1,700 transcript segments are ingested, normalized for punctuation, and chunked in under 100 milliseconds."*

### **Step 2: Grounded Q&A with Timestamp Video Seeking**
* **Select Strategy:** `fixed_token` (Top-$k=4$)
* **Ask Question:** `What distinguishes pre-training from fine-tuning?`
* **What to demonstrate:**  
  1. The LLM response cites numbered bracket citations `[1]`.
  2. Click the `Seek to 17:23` interactive timestamp button.
  3. Show the embedded video player jumping straight to Karpathy's explanation at `17:23` (even overcoming the raw ASR caption error *"fine chaining stage"*).

### **Step 3: Closed-World Guardrail (Adversarial Negative Rejection)**
* **Ask Question:** `Which treaty concluded the Napoleonic Wars in 1815?`
* **What to demonstrate:**  
  *"The system returns strictly `NOT_IN_VIDEO`. Across all 26 adversarial out-of-domain negative probes in our benchmark, our closed-world prompt achieved a 100% negative rejection rate, completely suppressing hallucinations on unanswerable topics."*

---

## 📊 3. Exact Numbers Cheat-Sheet (Reconciled with Paper)

| Finding / Metric | Exact Benchmark Value | Viva Defense Explanation |
| :--- | :--- | :--- |
| **Fixed vs. Semantic (Table I)** | `fixed_token`: **8.5% Hit@1**<br>`semantic`: **6.2% Hit@1** | Semantic chunking does not beat fixed-token; bootstrap 95% CIs overlap ([4%, 14%] vs [2%, 11%]). |
| **McNemar Significance Test** | **$p = 0.5078$** (Two-sided exact) | Paired test on 9 discordant pairs (6 fixed-only vs 3 semantic-only) proves no statistically significant gain ($p > 0.05$). |
| **Preprocessing Overhead (Table IV)** | **160× slower chunking step**<br>**4.4× total indexing latency** | Semantic chunking takes 12.80s vs 0.08s for fixed-token chunking per lecture hour. |
| **Search Space Collapse (Table III)** | Short ($\le$35m): **23% Hit@1**<br>Long ($>$35m): **3–6% Hit@1** | Lexical dispersion causes flat dense cosine similarity to match secondary topic mentions. |
| **Hierarchical Recovery (Algorithm 2)** | Hit@1 jumps from **6% to 18%**<br>Med error drops **1012s $\rightarrow$ 264s** | 2-stage indexing (5-min macro blocks $\rightarrow$ micro chunks) reduces search space and eliminates secondary collisions. |
| **Negative Rejection (Table V)** | **100%** (26 / 26 probes) | Closed-world prompt strictly returns `NOT_IN_VIDEO` on out-of-domain queries. |
| **Answerable Rejection (Table V)** | **53.3%** (16 / 30 rejected) | High refusal rates on answerable queries occur when top-$k$ retrieval misses the target window (Tier C: 10/10). |
| **Supported $\neq$ Correct (Table V)** | **100% supported** (14/14)<br>**7% gold overlap** (1/14) | LLM-as-a-judge confirms answers are grounded in retrieved chunks, but chunks themselves missed gold window in Tier B. |

---

## 🎯 4. Top 10 Tough Viva Questions & Winning Answers

#### **Q1: Why did Semantic Chunking fail to outperform simple Fixed-Token chunking?**
> **Answer:** *"In spoken lecture audio, natural acoustic pauses, conversational disfluencies ('um', 'you know'), and rhetorical questions create artificial local cosine similarity drops between consecutive sentence embeddings. This triggers premature chunk splits that fragment unified pedagogical explanations, giving it no advantage over fixed-token sliding windows ($p = 0.508$)."*

#### **Q2: Why did Tier C (NPTEL / Indian Accented) have a 100% rejection rate in end-to-end evaluation?**
> **Answer:** *"Under top-4 fixed-token retrieval, Tier C Hit@4 is 30% (70% misses). In our pseudo-random sample of 10 answerable queries (seed 42), the retrieved chunks lacked the necessary factual context to answer. Because our closed-world guardrail enforces strict non-hallucination, the LLM correctly abstained with `NOT_IN_VIDEO`. This highlights an essential insight: high refusal rates can mask upstream retrieval failure."*

#### **Q3: What is the 'Search Space Expansion Effect'?**
> **Answer:** *"In short lectures ($\le$35 min), retrieval achieves 23% Hit@1. But in long technical lectures ($>$35 min, up to 164 min), professors repeat core terms like 'gradient', 'loss', or 'weights' dozens of times across recaps and proofs. Flat dense retrieval collides with these secondary mentions, collapsing top-1 accuracy to 3–6% and expanding median error to $\sim$15 minutes."*

#### **Q4: How does your Two-Stage Hierarchical Video Retriever solve this?**
> **Answer:** *"Algorithm 2 partitions the lecture into coarse 5-minute macro blocks ($T_{\text{macro}}=300$s). Stage 1 retrieves the top-2 macro blocks representing the broad topical chapter (10 minutes total). Stage 2 only evaluates micro-chunks inside those active blocks. This recovered Hit@1 from 6% to 18% and cut median boundary error by 74%."*

#### **Q5: What is the difference between 'Supported' and 'Correct' in Table V?**
> **Answer:** *"Our LLM-as-a-judge found that 100% (14/14) of generated answers were fully supported by the cited chunks—meaning the model never hallucinated beyond its prompt. However, only 1 of the 14 cited a chunk that actually overlapped the human gold window. In Tier B, answers were grounded in off-target lecture segments mentioning related terms. Thus, guardrails prevent parametric confabulation but cannot detect off-target retrieval."*

#### **Q6: Why did you choose MedVidQA metrics over standard BLEU or ROUGE?**
> **Answer:** *"Text n-gram metrics like BLEU or ROUGE only measure lexical text overlap, ignoring whether the user was pointed to the right video timestamp. MedVidQA uses Intersection-over-Union (IoU) recall thresholds ($R@1 \text{ @ IoU} \ge 0.3/0.5/0.7$) and boundary start error, directly evaluating temporal localization precision."*

#### **Q7: What vector database and embeddings did you use, and why?**
> **Answer:** *"We used `all-MiniLM-L6-v2` (384-dimensional dense vectors) and Qdrant in local disk-backed embedded mode (`path='qdrant_data'`). This processes 2,450 words/second on standard laptop CPUs, allowing fully offline indexing of a 1-hour lecture in under 5 seconds without cloud server costs."*

#### **Q8: How did you handle transcripts with missing punctuation (The Punctuation Trap)?**
> **Answer:** *"Raw YouTube ASR streams often produce continuous unpunctuated text. We implemented a sentence boundary normalization pass in `transcript.py` that normalizes whitespace, splits run-on clauses using heuristic and NLP boundary rules, and associates continuous $[t_{\text{start}}, t_{\text{end}}]$ timestamps to every chunk."*

#### **Q9: What are the main limitations of your study?**
> **Answer:** *"We honestly disclose six limitations in Section VIII: (1) Gold spans were authored by a single domain expert; (2) Silent blackboard equations are missed by audio transcripts; (3) Macro-block embeddings use standard 256 word-piece truncation; (4) Chunks up to 200 words approach model truncation limits; (5) Sub-30s windows cause semantic fragmentation; and (6) Video duration and acoustic tier are partially confounded with modest per-tier samples."*

#### **Q10: What is the future scope of this research?**
> **Answer:** *"Our immediate next phase integrates multimodal slide and blackboard visual frame embeddings (using CLIP / SigLIP) to capture unvocalized mathematical equations, alongside hybrid BM25 lexical indexing to eliminate secondary keyword collisions."*
