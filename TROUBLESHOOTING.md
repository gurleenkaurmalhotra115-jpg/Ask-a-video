# VidQuery: Defense Day Troubleshooting & Offline Playbook

This document is your **emergency reference guide** for common real-world issues that can arise during live project evaluations, viva presentations, or classroom demonstrations.

---

## 1. Classroom Wi-Fi or Internet Fails (100% Offline Mode)

### What Still Works Offline:
- **All 13 Benchmark Lectures:** All transcripts are frozen in `data/transcripts/captions/*.json`. They load in **under 0.1 seconds from disk** without touching YouTube.
- **Chunking Engine:** All 4 chunking strategies (`fixed_token`, `sentence`, `semantic`, `time_window`) run locally in memory.
- **Vector Indexing & Search:** `sentence-transformers/all-MiniLM-L6-v2` and Qdrant run **100% locally on your laptop's CPU** with zero network egress.
- **Disk-Cached QA:** Any previously asked benchmark queries in `qa_log.jsonl` are loaded from disk cache without making Gemini API calls.
- **Slide Deck & PDF:** [`presentation.html`](presentation.html) and [`paper.pdf`](paper.pdf) require zero internet.

### What Needs Mobile Hotspot:
- Only *new, previously unasked questions* sent to the Gemini 3.5 Flash Lite LLM require internet.
- **Quick Fix:** Turn on your phone's personal hotspot if the university Wi-Fi drops.

---

## 2. Port Collision: "Port 8502 is already in use"

If you see an error like `[WinError 10048] Only one usage of each socket address is normally permitted`:

### Solution A: Launch on a different port
```powershell
.\venv\Scripts\streamlit.exe run app.py --server.port 8503
```
Then open `http://localhost:8503` in your browser.

### Solution B: Terminate the existing Streamlit process
In PowerShell:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*streamlit*" } | Stop-Process -Force
```
Or in Command Prompt:
```cmd
taskkill /F /IM python.exe
```

---

## 3. YouTube IP Rate-Limit Warning ("Could not retrieve captions")

### Why It Happens:
YouTube aggressively blocks automated scraping requests from residential and university IP addresses via `youtube-transcript-api`.

### What to Say to the Examiner:
> *"Sir/Ma'am, commercial video platforms like YouTube employ automated IP rate-limiting against web scrapers. In production, systems use official Google Cloud OAuth credentials or dedicated proxy rotators. For our benchmark evaluation, we pre-indexed and frozen 13 diverse lecture transcripts (~20,000 segments) across three acoustic tiers into local immutable JSON cache files. This guarantees deterministic reproducibility and allows our entire system to operate 100% offline."*

### How to Demonstrate Instantly:
1. In the Streamlit UI, pick any of the **13 pre-indexed benchmark lectures** from the dropdown above the URL input:
   - *3Blue1Brown - What is a Neural Network? (aircAruvnKk)*
   - *Andrej Karpathy - Intro to Large Language Models (zjkBMFhNj_g)*
   - *Stanford CS229 Andrew Ng (jGwO_UgTS7I)*
   - *MIT Strang - Linear Algebra (J7DzL2_Na80)*
   - *NPTEL IIT Madras ML (YLczBemZpaE)*
2. Click **Fetch transcript**. It will load immediately without making any YouTube network requests.

---

## 4. Examiner Asks: "Where are the actual vectors stored?"

### Quick Answer:
> *"Our vectors are stored in a local, embedded Qdrant collection on disk under `qdrant_data/`. We do not rely on a separate Docker container or cloud database server."*

### Live Inspection Command:
To show the examiner the local database files:
```powershell
Get-ChildItem qdrant_data
```
To show the cosine vector dimensions and collections in Python:
```powershell
.\venv\Scripts\python.exe -c "from vectorstore import get_client, list_indexed; c = get_client(); print(list_indexed(c))"
```

---

## 5. Fast Recovery Checklist (If App Closes Unexpectedly)

1. **Double-click:** [`launch_demo.bat`](launch_demo.bat) on your desktop.
2. **If that fails, run in terminal:**
   ```powershell
   cd "c:\Users\gurle\OneDrive\Desktop\cm project"
   .\venv\Scripts\streamlit.exe run app.py --server.port 8502
   ```
3. **If Python says module missing:**
   ```powershell
   .\venv\Scripts\pip.exe install -r requirements.txt
   ```
4. **Open your presentation backup:**
   Double-click [`presentation.html`](presentation.html) in your browser. It runs completely offline without any servers.
