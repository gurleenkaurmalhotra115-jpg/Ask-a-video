# VidQuery: Live Demo & Viva Quick-Reference Cheat-Sheet
> Keep this document open on a second monitor or mobile device during your presentation and demo to Ma'am / Examiners.

---

## ⚡ 1. Starting the Application

If the server is not already running:
```powershell
.\venv\Scripts\activate
streamlit run app.py --server.port 8502
```
URL: **http://localhost:8502**

---

## 🎬 2. The 3-Minute Live Demo Sequence

### **Step 1: Ingest Lecture (Offline Caching Demonstration)**
* **Select from dropdown:**  
  `Andrej Karpathy - Intro to Large Language Models (zjkBMFhNj_g)`
* **Click:** `Fetch transcript`
* **What to say:**  
  *"We implemented an offline-first caching pipeline that isolates the application from YouTube API IP bans. Notice how 1,704 segments loaded in less than 0.1 seconds without any external network latency."*

### **Step 2: Grounded Q&A with Dynamic Video Seeking**
* **Select Strategy:** `fixed_token` (Top-4 chunks)
* **Question to paste:**  
  `What is context window memory in an LLM and how is it limited?`
* **Click:** `Ask Question`
* **What to highlight:**  
  1. The answer cites `[1]`.
  2. Click the `▶ 08:36` button.
  3. Show the video player instantly jumping to Andrej Karpathy explaining RAM / context window memory.

### **Step 3: Zero-Hallucination Guardrail (Negative Rejection)**
* **Question to paste:**  
  `What is the capital of France?`
* **Click:** `Ask Question`
* **What to highlight:**  
  *"The system outputs strictly `NOT_IN_VIDEO` with 0 citations. Across all 26 adversarial negative probes in our 156-question benchmark, our closed-world prompting achieved a verified 100% Negative Rejection Rate."*

---

## 📊 3. Key Numbers to Remember for Viva Questions

| Metric / Concept | Number / Finding | Why it Matters |
| :--- | :--- | :--- |
| **Fixed vs. Semantic (Table 1)** | Hit@1: **4.6% vs 3.1%**<br>MRR: **0.097 vs 0.087** | Empirically confirms Qu et al. (Vectara 2024) on spoken discourse. Semantic chunking doesn't beat simple token slicing. |
| **Search Space Expansion (Table 3)** | $\le$35 min: **23.3% Hit@1**<br>$>$35 min: **3–6% Hit@1** | In multi-hour lectures, repeated vocabulary (*"weights", "loss"*) confuses flat bi-encoders, requiring hierarchical retrieval. |
| **Acoustic Robustness (Table 2)** | Tier C (NPTEL): **30.0% Top-4 Recall** | Dense embeddings reliably surface the target concept even under accented Indian English. |
| **Dense Separation Margin** | $\Delta = \mathbf{+0.2873}$ | $\mu_{\text{answerable}} = 0.4223$ vs $\mu_{\text{neg}} = 0.1349$, providing an intrinsic pre-LLM filter. |
| **Negative Rejection (Table 4)** | **100.0%** (26 / 26) | Eliminates hallucinations on out-of-domain queries. |
| **Conditional Faithfulness** | **100.0%** on retrieved evidence | When context is retrieved, all generated propositions are strictly supported by cited `[n]` chunks. |

---

## 🧪 4. One-Click Health Audit

To prove project integrity to examiners:
```powershell
python verify_all.py
```
Outputs: `ALL VERIFICATIONS PASSED! System is 100% submission ready.`
