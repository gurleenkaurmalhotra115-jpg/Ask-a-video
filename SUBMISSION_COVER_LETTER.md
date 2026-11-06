# Conference Submission Cover Letter & Executive Summary

**Manuscript Title:** Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise  
**Author:** Gurleen Kaur Malhotra  
**Author Affiliation:** Department of Information Technology, Maharaja Surajmal Institute of Technology (MSIT), Guru Gobind Singh Indraprastha University (GGSIPU), New Delhi, India  
**Author Contact:** `gurleen.research@msit.in`  
**Target Tracks / Workshops:**
* Retrieval-Augmented Generation / Information Retrieval
* NLP for Educational Applications (e.g., BEA, ACL/EMNLP Workshops)
* Video Question Answering & Temporal Grounding (IEEE / ACM Proceedings)

---

## Cover Letter to Program Committee & Area Chairs

Dear Program Committee and Area Chairs,

We are pleased to submit our original research paper, *"Evaluating Chunking Strategies and Temporal Grounding in Long-Form Educational Video RAG Under Acoustic Noise,"* for consideration as a conference paper.

### Context and Timeliness
While Retrieval-Augmented Generation (RAG) has rapidly matured across text-centric document collections, educational institutions increasingly rely on continuous, unpunctuated spoken media (e.g., lectures from MIT OpenCourseWare, Stanford Online, and India's NPTEL). Navigating multi-hour technical video lectures requires precise **temporal grounding**—not only answering student questions, but generating clickable timestamps that seek to the exact second in the lecture.

### Empirical Contributions & Key Findings
1. **Empirical Validation of Qu et al. (Vectara, 2024) on Spoken Discourse:**  
   Prior literature debated whether embedding-based semantic boundary chunking justified its quadratic computational overhead over fixed-token splitting. We conduct an empirical investigation across 13 full-length STEM lectures (~20,000 segments) and 156 validated queries using the **MedVidQA (ACL 2022)** temporal intersection framework ($R@1 \text{ @ IoU} \ge 0.3/0.5/0.7$). We find that `semantic` chunking trails behind simple `fixed_token` partitioning (Hit@1: **3.1% vs 4.6%**, MRR: **0.087 vs 0.097**), confirming that semantic splitters over-fragment on conversational shifts in speech without providing higher localization precision.

2. **Formalization of the Search Space Expansion Effect:**  
   We quantify the performance cliff between short lectures ($\le$35 min) and extended classroom lectures ($>$35 min). While video-native temporal partitioning achieves **23.3% Hit@1** on short videos, multi-hour lectures introduce high lexical repetition of foundational keywords (*"loss"*, *"weights"*, *"vectors"*), causing flat dense top-1 localization to compress to 3–6%. This provides architectural evidence for the necessity of two-stage hierarchical indexing.

3. **Acoustic Robustness Across Accents (NPTEL):**  
   We evaluate across three pedagogical and acoustic tiers (Curated Studio, US Classroom, and Indian-accented English from NPTEL). Despite reverberant acoustics and non-native phonetic features, dense bi-encoders reliably preserved **30.0% Top-4 Recall** in Tier C.

4. **Zero-Hallucination Barrier:**  
   We demonstrate a pre-LLM dense separation delta of $\Delta = +0.2873$ between answerable queries and negatives. With closed-world citation constraints, our system achieves a **100% Negative Rejection Rate** across all 26 adversarial out-of-domain probes, eliminating hallucinations.

### Compliance and Integrity
* This work is original and has not been concurrently submitted to another venue.
* The paper is formatted in standard **IEEE two-column conference format** (`paper.tex`).
* The open-source system, 156-question benchmark, and reproduction scripts are released in the accompanying artifact bundle.

We thank the reviewers for their time and valuable feedback.

Sincerely,  
**Gurleen Kaur Malhotra**  
Department of Information Technology  
Maharaja Surajmal Institute of Technology, GGSIPU, New Delhi
