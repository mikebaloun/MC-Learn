# MC-Learn: Accelerating Transformer Training with Monte Carlo Sampling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikebaloun/MC-Learn/blob/main/Monte_Carlo_Learn.ipynb)

**Reduced Transformer training time by nearly 3×** through a new adaptive sampling method that focuses computation on the most valuable examples.  
In initial experiments on **AG News**, MC-Learn achieved a **2.83× speedup** with about a **6% accuracy trade-off**, demonstrating the trade-off between efficiency and accuracy.

---

## 🚩 Motivation

Training large language models is expensive.  
Standard fine-tuning treats every example equally — spending the same compute on redundant, easy samples as on difficult, high-variance ones.  

MC-Learn changes this by **learning where to spend compute**, cutting down wasted training time.

---

## 🧩 How It Works

Instead of processing the entire dataset each step:

1. **Score** – A lightweight surrogate head scores a candidate pool for difficulty.  
2. **Select** – A smaller, high-value batch is drawn (with replay regularization).  
3. **Train** – The model updates on this batch with a stable gradient estimate.  
4. **Adapt** – Sampling policy and budget adjust dynamically based on runtime and variance signals.  

This creates a **compute-efficient training loop**: spend more on the hardest, most useful data; spend less on redundant data.

---

## ⚙️ System Components

### 🧠 Policy Learner  
Learns a sampling strategy online:
- **Class Weights** – allocate more budget to classes with higher gradient variance  
- **Sampling Temperature (τ)** – controls sharpness of within-class selection  
- **Replay Share (ρ)** – mixes in a replay buffer to prevent forgetting  
- **ESS Guard** – monitors effective sample size for stability  

### ⚖️ Budget Solver  
Keeps training within a fixed compute budget:
- Tracks real **per-example costs** (scoring, surrogate, full forward)  
- Solves for optimal batch size *M* and sampling probabilities *q*  
- Ensures wall-clock training stays within the target budget  

### 👷 Sampler  
Builds the actual training batch:
- Scores candidates with the surrogate head  
- Uses **soft top-k + replay union** to keep both hard and easy examples  
- Computes **clipped importance weights** for unbiased, stable training  

### 🔄 Trainer  
Runs the training loop with a **variance-reduced estimator**:

- Every example goes through the **surrogate head** for a fast but approximate loss  
- A smaller subset also runs the **full forward pass**, correcting bias  
- **Importance weights** and **inclusion probabilities** ensure statistical validity  

This makes training **much faster** while keeping gradient estimates **stable and nearly unbiased**.

---

## 📊 Results

### Experimental Setup
- **Model:** DistilBERT (`distilbert-base-uncased`)  
- **Dataset:** AG News (120k training samples, 4-class classification)  
- **Environment:** Google Colab (T4 GPU)  
- **Runs:** 1 epoch equivalent, averaged over 3 seeds (42/43/44)  

### Performance (AG News)

These results are from a **single benchmark task** (AG News).  
They should be read as a **case study**, not a universal guarantee.

| Run      | Accuracy (Mean ± Std)   | Time (s) (Mean ± Std) | Speedup |
|----------|-------------------------|-----------------------|---------|
| Baseline | 0.9309 ± 0.0004         | 316.3 ± 16.4          | 1.00×   |
| MC-Learn | 0.8698 ± 0.0008         | 111.7 ± 2.3           | 2.83×   |

**Takeaway:** On AG News, MC-Learn trains ~2.8× faster with a ~6% absolute accuracy drop.

---

## 🔍 Diagnostics

MC-Learn includes built-in analysis tools:  
- **Confusion Matrix** – shows which classes are most often confused  
- **Difficulty Score Distributions** – highlights mismatches between the surrogate scorer and actual difficulty  

These tools help identify when the scoring mechanism is under- or over-confident.

---

## 📚 Background

MC-Learn builds on established ideas:  
- **Importance Sampling & Variance Reduction** – from Monte Carlo methods  
- **Curriculum & Active Learning** – focusing on more informative data  
- **Efficient Transformer Training** – complements pruning, quantization, and distillation  

---

## 🚀 How to Run

1. Click the **Colab badge** at the top  
2. Enable GPU: **Runtime → Change runtime type → GPU**  
3. Run all cells to:  
   - Train a baseline run  
   - Train MC-Learn with adaptive sampling  
   - Output metrics and diagnostics  

---

## 📌 Notes

- Automatically adapts presets for varying GPUs (e.g. T4, A100)  
- Importance weights clipped at `w_clip = 10.0` for stability  
- Designed for reproducibility: fixed seeds and evaluation protocol  
- Results reported **only on AG News**; performance may differ elsewhere  
- Future work: test on SST-2, IMDB, MNLI and scale to larger models  
- **Best suited for:** rapid prototyping, ablation studies, and compute-limited settings  
- **Not intended** as a drop-in replacement for full fine-tuning when maximum accuracy is the goal  

---
