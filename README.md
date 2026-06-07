# Grokking with Attention Heads — Mechanistic Interpretability

A study on the grokking phenomenon in transformers, extending Neel Nanda's work by introducing positional embeddings and bias terms, and investigating their effects on generalization.

**Authors:** Shreeyans Arora, Muskaan Maurya

---

## What is Grokking?

Grokking is a phenomenon where a neural network first memorizes the training data, and then, suddenly generalizes to unseen data after further training. The model achieves a balance between memorization and generalization.

This project is based on the paper [*"Progress Measures for Grokking via Mechanistic Interpretability"*](https://arxiv.org/abs/2301.05217) by Neel Nanda.

---

## Problem Statement

Neel Nanda's original experiment trained a vanilla transformer (without positional embeddings or bias) to perform modular addition. The model eventually generalized, showing a clean drop in test loss after an extended period of memorization.

**Our question:** What happens when we add positional embeddings and bias to this setup? Do these components change how or whether the model generalizes?

We also observed unexpected behavior: repeated spikes in log loss instead of a clean dip, and investigated what causes them.

---

## Experimental Setup

### Task
- **Input:** Three tokens — `a`, `b`, and `=`, where `a` and `b` are integers from 0 to 112.
- **Output:** `(a + b) mod 113`
- **Training duration:** 10⁵ epochs

### Model Architecture

| Parameter | Value |
|---|---|
| Embedding dimension | 128 |
| Attention dimension | 128 |
| Number of heads | 4 |
| MLP hidden size | 512 |
| Non-linearity | ReLU |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1 |

The model uses GPT-style embeddings: trainable token embeddings combined with positional embeddings derived from *Attention Is All You Need*.

### Our Changes vs. Neel Nanda's Setup
- Added **positional embeddings** (sinusoidal, non-trainable)
- Added **bias terms**
- Tested both **float32** and **float64** precision

---

## The Four Experimental Setups

Each setup was run on 10 random seeds: `[1, 21, 32, 42, 55, 100, 234, 456, 856, 12]`

| Setup | dtype | Bias |
|---|---|---|
| Setup 1 | float32 | No |
| Setup 2 | float32 | Yes |
| Setup 3 | float64 | No |
| Setup 4 | float64 | Yes |

---

## Results

### Setup 1 — float32, no bias
- Loss spikes appeared in all seeds, but spike count varied across seeds.
- Most seeds (1, 21, 32, 42, 55, 100, 234, 856) showed a smooth overall trend.
- Seeds 12 and 456 showed an aggressive spike pattern. The log loss did not drop to zero for these seeds.

### Setup 2 — float32, with bias
- Loss spikes appeared in all seeds. Spike count increased gradually with higher seed values.
- All seeds showed a smooth downward trend overall.
- Seed 12 was an outlier — it had more spikes than expected.

### Setup 3 — float64, no bias
- **No spikes** were observed in any seed. This matches Neel Nanda's original result.
- Train and test loss curves ran parallel with a consistent gap between them.
- Seed 456 showed a slight increase in the train-test gap.

### Setup 4 — float64, with bias
- Most seeds (12, 42, 100, 234, 856) showed no spikes — similar to Setup 3.
- Seed 1 showed unusual behavior in the middle of training.
- Seeds 32 and 55 showed a similar trend to each other, with a small difference in the train-test gap.
- Seed 456 initially followed the normal trend, but spikes appeared later.
- Seed 21 showed no spikes with a slight increase in the train-test gap.

---

## Hypotheses for the Spike Behavior

We observed spikes in log loss in all float32 setups and some float64+bias setups. Three possible causes were considered:

1. **Weight decay** — High weight decay values may force the model to effectively re-initialize its weights, disrupting the training trajectory.

2. **Positional embedding mismatch** — Token embeddings are trainable, while positional embeddings are not. The optimizer handles them differently, which could create instability.

3. **Numerical precision (dtype)** — float64 increases arithmetic precision. This makes the model more sensitive to small gradient changes. Switching to float64 eliminated spikes when bias was absent, but only reduced them when bias was present.

---

## Weight Decay Experiments

All weight decay experiments were done on float32 with bias, using seed=1.

| Weight Decay | Behavior |
|---|---|
| 0 | Model memorizes well but generalizes very slowly |
| 0.0001 | Smooth convergence, moderate spikes |
| 0.1 | More irregular spikes begin to appear |
| 0.005 | Fewer but larger spikes |
| 0.5 | Repeated spikes, train and test loss both diverge |
| 1 | Aggressive repeated spikes throughout training |

Higher weight decay values increase the frequency and severity of spikes.

---

## Positional Embedding Ablation

We tested with and without positional embeddings under the same conditions. The results were identical in both cases. This shows that **positional embeddings do not cause the spikes**.

---

## Detailed Analysis of Spikes

To understand the spikes, we compared model state at the **peak** of a spike (highest log loss) vs. the **trough** (lowest log loss).

**Cosine similarity of activations between peak and trough:**
- Token embedding layer: **0.76**
- Final embedding (after positional embedding): **0.99**
- After MLP layer: **0.88**

**Cosine similarity of parameters between peak and trough:**
- All layers except token embedding: **< 0.4** — the parameters change significantly during a spike.

**Cosine similarity after PCA (top 10 dimensions):**
- Most layers: **> 0.9**
- MLP layer (mlp_0): **0.8**
- The 10th principal component of all layers: **< 0.1**

**ReLU activations:**
- At spike peak: ~**30%** of all activations (across 12,769 input points) were zero.
- At spike trough: ~**10%** of activations were zero.
- No layer other than ReLU had zero-valued activations or parameters, which rules out **vanishing gradients**.

---

## Attention Head Findings

When the model generalizes, the attention values between tokens `a→b` and `b→a` overlap each other. This is consistent with Neel Nanda's result that the model learns to ignore positional order for modular addition.

Interestingly, even when log loss is high (during a spike), the average attention values between `a,b` and `b,a` still overlap. This suggests the model has not forgotten its generalized representation during the spike.

The attention values follow a **sinusoidal pattern** over training, indicating structured, oscillatory dynamics in how the model attends to inputs.

---


## References

- Neel Nanda et al., [*"Progress Measures for Grokking via Mechanistic Interpretability"*](https://arxiv.org/abs/2301.05217), 2023.
- Vaswani et al., [*"Attention Is All You Need"*](https://arxiv.org/abs/1706.03762), 2017.
