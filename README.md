# MC-Learn: Accelerating Transformer Training with Monte Carlo Sampling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikebaloun/MC-Learn/blob/main/Monte_Carlo_Learn.ipynb)

MC-Learn is an adaptive sampling algorithm designed to reduce the wall-clock time and computational cost of fine-tuning Transformer models. By selectively training on the **most informative examples**, it achieves a ~2.8× speedup in training at the cost of about a 6% absolute accuracy drop.

---

## Motivation

Training large language models is expensive.  
Standard fine-tuning wastes resources by treating **every example equally** — spending the same compute on easy, redundant data as on difficult, high-variance samples.

MC-Learn changes this by **learning where to spend compute**.

---

## How It Works

Instead of processing the full dataset every step:

1. **Score:** A lightweight surrogate head scores a candidate pool for difficulty.  
2. **Select:** A smaller, high-value batch is drawn (with replay regularization).  
3. **Train:** The model updates on this batch with a near-unbiased, low-variance gradient estimate.  
4. **Adapt:** Sampling policy and budget update dynamically based on runtime measurements and variance signals.

This creates a **compute-efficient training loop**: spend more on the hardest, most useful data; spend less on redundant data.

---

## ⚙️ System Components

### Policy Learner
Learns a sampling strategy online:
- **Class Weights:** Allocate more budget to classes with higher gradient variance.  
- **Sampling Temperature (τ):** Control sharpness of within-class selection.  
- **Replay Share (ρ):** Mix in a replay buffer to regularize and prevent forgetting.  
- **ESS Guard:** Monitors effective sample size to keep importance weighting stable.

### Budget Solver
Allocates compute under a fixed budget:
- Tracks **measured per-example costs** of scoring, cheap head, and full forward.  
- Solves for optimal batch size *M* and inclusion probabilities *q*.  
- Ensures wall-clock training stays within target budget ratio.

### Sampler
Builds the actual training batch:
- Scores a candidate pool with the surrogate head.  
- Uses **soft top-k + replay union** to keep both hard and easy examples.  
- Computes clipped **importance weights** for unbiasedness and variance control.

### Trainer
Runs the training loop with a **control variate estimator**.  

In practice:
- Every sampled example is passed through the **surrogate head**, producing a fast but approximate loss.  
- A smaller subset of examples is also run through the **full forward pass**. The difference between full and surrogate losses is scaled up to correct for subsampling.  
- **Importance weights** and **inclusion probabilities** ensure the estimator is statistically valid.  

This combination makes the training **much faster**, while keeping the gradient estimate **nearly unbiased**.

---

## Results

### Experimental Setup
- **Model:** `distilbert-base-uncased` (4-class classifier head)  
- **Dataset:** AG News (120k training samples)  
- **Task:** Topic classification (World, Sports, Business, Sci/Tech)  
- **Environment:** Google Colab T4 GPU  
- **Runs:** 1 epoch equivalent, averaged over 3 seeds (42/43/44)  

### Performance

| Run      | Accuracy (Mean ± Std)   | Time (s) (Mean ± Std) | Speedup |
|----------|-------------------------|-----------------------|---------|
| Baseline | 0.9309 ± 0.0004         | 316.3 ± 16.4          | 1.00×   |
| MC-Learn | 0.8698 ± 0.0008         | 111.7 ± 2.3           | 2.83×   |

**Takeaway:** MC-Learn trains **~2.8× faster** with only a **6.1% absolute accuracy drop**.  
This is ideal for rapid prototyping and compute-constrained settings.

*Note: Comparison is compute-matched (wall-clock), not loss-matched. MC-Learn uses EMA distillation for stability.*

---

## Diagnostics

MC-Learn includes built-in analysis tools:
- **Confusion Matrix**: visualize which classes are confused.  
- **Difficulty Score Distributions**: detect mismatches between scorer and true difficulty (“smoking gun” analysis).  

These help identify when the surrogate scorer is under- or over-confident.

---

## Background

MC-Learn builds on established concepts:
- **Importance Sampling & Control Variates** — variance reduction in Monte Carlo estimators.  
- **Curriculum & Active Learning** — focusing compute on harder or more informative examples.  
- **Efficient Transformer Training** — complements pruning, quantization, and distillation.  

---

## How to Run

1. Click the **Colab badge** at the top.  
2. Enable GPU: **Runtime → Change runtime type → GPU**.  
3. Run all cells.  
4. The script will:
   - Train a baseline run  
   - Train MC-Learn with adaptive sampling  
   - Output metrics and diagnostics  

---

## Notes
- Automatically adapts presets for fast GPUs (e.g. A100 with bf16).  
- Importance weights are clipped at `w_clip = 10.0` to ensure stability.  
- Designed for reproducibility: controlled seeds and fixed evaluation protocol.  

---
