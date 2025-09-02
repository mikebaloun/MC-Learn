# Accelerating Transformer Training with Monte Carlo Sampling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikebaloun/MC-Learn/blob/main/Monte_Carlo_Learn.ipynb)

This project, titled MC-Learn, is an implementation of an adaptive sampling algorithm designed to reduce the time and computational cost of fine-tuning Transformer models. By intelligently selecting the most informative data points during training, this method achieves performance comparable to traditional training methods in a fraction of the time.

## The Problem

Training large language models is a resource-intensive process. Standard training methods treat all data points equally, expending the same computational effort on low-loss, redundant examples as on high-loss, informative ones.

## The Solution: How It Works

This algorithm employs an adaptive sampling strategy based on Monte Carlo methods to improve training efficiency. Instead of processing every data point, a lightweight surrogate model first evaluates the difficulty of a candidate pool of examples. The system then constructs a smaller, high-value training batch from this pool, focusing the computational budget on the examples most beneficial for model improvement.

## Algorithm Architecture: A Deeper Look

The system is composed of several key modules that work in concert to optimize the training process.

### 🧠 The Policy Learner

This is the core decision-making component of the system. It continuously learns and updates a sampling policy to identify which data provides the most informative gradient signal at any given point in training. It manages this by adapting three key parameters:
1.  **Class Weights:** Dynamically allocates more budget to entire data classes where the model exhibits higher gradient variance.
2.  **Sampling Temperature:** Controls the sharpness of the sampling distribution within a class, balancing between exploiting the most difficult examples and exploring a wider variety of data.
3.  **Replay Share:** Determines the proportion of data drawn uniformly from a replay buffer, which acts as a regularizer to ensure the model maintains general knowledge and does not overfit to the adaptively-sampled "hard" examples.

### ⚖️ The Budget Solver

This is the resource allocation module. Its function is to ensure the training process adheres to a predefined computational budget (e.g., a target percentage of the baseline training time). It continuously measures the wall-clock cost of its core operations and uses these metrics to solve for the optimal batch size that minimizes variance subject to the budget constraint.

### 👷 The Sampler

This is the data selection module. It executes the strategy defined by the `Policy Learner` and `BudgetSolver`. It scores a large candidate pool of data, then constructs the final, high-value batch according to the learned policy for the `Trainer` to process.

### ⚙️ The Trainer

This is the orchestration engine that manages the end-to-end training loop. It coordinates the other modules and, most critically, computes the loss using a **control variate** gradient estimator. This statistical technique ensures that the gradient calculated from the small, biased subset of data is an unbiased and low-variance estimate of the true gradient over the entire dataset.

## Results & Analysis

### Experimental Setup

* **Model:** **DistilBERT** (`distilbert-base-uncased`), a lighter and faster version of BERT, chosen for its efficiency.
* **Dataset:** **AG News**, a standard benchmark for text classification, consisting of 120,000 training samples.
* **Task:** 4-class news topic classification (World, Sports, Business, Sci/Tech).
* **Baseline:** The "Baseline" run represents standard model fine-tuning for one full epoch on the entire training dataset.

### Performance

The method demonstrates a clear and significant trade-off between training speed and final model accuracy.

| Run      | Accuracy | Time (s) | Speedup |
|----------|----------|----------|---------|
| Baseline | 0.9307   | 321.7s   | 1.00x   |
| MC-Learn | 0.8688   | 108.5s   | 2.96x   |

The algorithm achieved a **2.96x speedup** in training time, a substantial improvement that can lead to significant cost savings and faster development cycles. This performance gain was accompanied by an absolute drop of **6.19%** in accuracy, highlighting the algorithm's effectiveness in scenarios where training speed is a higher priority than achieving maximum accuracy (e.g., rapid prototyping).

## How to Run

1.  Click the "Open in Colab" badge at the top of this page.
2.  The notebook will open in Google Colab.
3.  Run the cells in the notebook to execute the code and see the results.
