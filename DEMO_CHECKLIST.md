# ReviewLens AI — Demonstration Checklist & Video Guide

This checklist details the demonstration sequence, local demo video verification, presentation script, and publication instructions for **SWYNEX Internship Task 4: Final AI Application**.

---

## 1. Local Demo Video Status & Verification

A screen recording of ReviewLens AI was verified on the local development workstation:
- **Local File Path**: `~/Desktop/Screen Recording 2026-10-18 at 8.45.15 PM.mov`
- **Duration**: ~23.54 seconds
- **Resolution**: 2816 × 1762 pixels
- **File Size**: 11.4 MB
- **Observed Contents**:
  - ReviewLens AI Streamlit UI loaded in Google Chrome (`localhost:8501`).
  - Input sentence: *"the laptop is good"* with aspect *"laptop"*.
  - Execution of aspect sentiment analysis yielding **POSITIVE** sentiment, class probability distribution, and heuristic metrics.

### Privacy & Publishing Recommendation:
> [!NOTE]
> The raw recording captures browser bookmarks and account indicators. If you wish to use this recording publicly, either:
> 1. Upload it to **Google Drive** or **YouTube (Unlisted)** with permissions set to "Anyone with the link can view".
> 2. Or re-record a 60–90 second clean browser window following the 7-step presentation script below.
>
> Paste your public video URL into the designated placeholder in `README.md` (Section I).

---

## 2. 7-Step Demonstration Sequence

Use this structured walkthrough for live evaluation or video presentation:

### Step 1: Project Introduction & Problem Framing (0:00 - 0:15)
- Introduce **ReviewLens AI** as an Aspect-Based Sentiment Analysis (ABSA) system developed for the SWYNEX Technologies Internship.
- Explain the core challenge: standard document-level sentiment models miss granular polarity differences across specific product features (e.g., loving a laptop's display while disliking its battery life).

### Step 2: Application Startup (0:15 - 0:25)
- Show the terminal and activate the virtual environment:
  ```bash
  source .venv/bin/activate
  streamlit run app.py
  ```
- Point out that the application runs locally and offline at `http://localhost:8501` without requiring external cloud API keys.

### Step 3: Positive Prediction Demonstration (0:25 - 0:40)
- In the Streamlit UI, choose the preset or type:
  - **Sentence**: `the laptop is very good`
  - **Aspect**: `laptop`
- Click **🚀 Analyze Aspect Sentiment**.
- Highlight:
  - **Predicted Sentiment**: `POSITIVE` (Green badge)
  - **Top Class Score**: `84.00%`
  - **Decision Margin**: `75.55%`
  - **Reliability Level**: `High`
  - Show the bar chart displaying the 4-class probability distribution.

### Step 4: Model Failure & Ambiguity Cases (0:40 - 1:00)
- Select the preset **Known Failure Case 1 — 'keyboard'**:
  - **Sentence**: `The keyboard feels cheap and uncomfortable.`
  - **Aspect**: `keyboard`
- Click **Analyze Aspect Sentiment**.
- Explain the real model limitation:
  - The linear TF-IDF model erroneously predicts `POSITIVE` (`42.33%`).
  - Demonstrate the **Intelligent Safeguard**: The decision margin drops to `9.70%`, triggering an automatic **Low / Ambiguous** warning with clear visual explanation.
- Select the preset **Missing Aspect Case**:
  - **Sentence**: `Great lightweight design and very durable.`
  - **Aspect**: `battery`
- Demonstrate how the system detects the absent aspect, falls back to prefix encoding, and flags a warning.

### Step 5: Heuristic Reliability Interpretation (1:00 - 1:15)
- Open the collapsible section: **"🛠️ Inspect Preprocessed Representation & Guidance Details"**.
- Explain why decision margin $\Delta = P(\text{top}_1) - P(\text{top}_2)$ matters:
  - Linear classifier softmax scores are uncalibrated distances.
  - The decision margin and aspect presence check prevent false overconfidence and guide human reviewers.

### Step 6: Automated Test Suite Execution (1:15 - 1:30)
- Switch to the terminal and execute the automated test suite:
  ```bash
  python -m unittest -v tests/test_reviewlens.py
  ```
- Demonstrate all 23 unit and integration tests passing in milliseconds (testing validation, aspect detection, error handling, margin calculation, and backward compatibility).

### Step 7: Final GitHub Repository & Documentation (1:30 - 1:45)
- Show the final GitHub repository: [Jyo-08/SWYNEX-AI-APPLICATION](https://github.com/Jyo-08/SWYNEX-AI-APPLICATION).
- Highlight the comprehensive README, clean directory structure, exclusion of raw datasets, and verified model checksum.

---

## 3. How to Upload & Link Your Demo Video

1. **Option A: Google Drive (Recommended for Quick Sharing)**
   - Open [Google Drive](https://drive.google.com).
   - Drag and drop `~/Desktop/Screen Recording 2026-10-18 at 8.45.15 PM.mov`.
   - Right-click the uploaded file $\rightarrow$ **Share** $\rightarrow$ Change General Access to **"Anyone with the link"** (Viewer).
   - Copy the shareable link.
   - Edit `README.md` Section I and replace the video link placeholder with your URL.

2. **Option B: YouTube (Unlisted)**
   - Go to [YouTube Studio](https://studio.youtube.com).
   - Click **Create** $\rightarrow$ **Upload video**.
   - Select the `.mov` recording.
   - Set visibility to **Unlisted** (viewable by anyone with the link, but not searchable publicly).
   - Copy the link and update `README.md`.

3. **Option C: GitHub Release Asset**
   - In `Jyo-08/SWYNEX-AI-APPLICATION`, click **Releases** $\rightarrow$ **Draft a new release**.
   - Tag: `v1.0.0`, Title: `ReviewLens AI v1.0.0 Final Release`.
   - Attach the video file as a release binary asset.
