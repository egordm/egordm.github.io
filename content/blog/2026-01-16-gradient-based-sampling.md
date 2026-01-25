---
title: "Gradient-Based Sampling (GOSS)"
date: 2026-01-16
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - algorithms
  - optimization
description: "How LightGBM speeds up training by keeping high-gradient samples and subsampling the rest. Plus row and column sampling strategies."
aliases:
  - "gb-part-6"
  - "gradient-based-sampling"
series: "Inside Gradient Boosting"
series_order: 6
---

> [!abstract] Inside Gradient Boosting, Part 6 of 9
> This series explains gradient boosting from first principles to advanced implementation details.
>
> **Previous:** [[gb-part-5|Depth-Wise vs Leaf-Wise Tree Growth]]
> **In this post:** Sampling strategies that speed up training while preserving accuracy.
> **Next:** [[gb-part-7|Exclusive Feature Bundling and Categorical Features]]

Building histograms is the dominant cost in gradient boosting. With a million samples, every tree must iterate over all of them to build gradient statistics. Can we do better?

The answer is **sampling**. Instead of using all samples to build each tree, we can train on a subset. This post explores two approaches: random subsampling and LightGBM's smarter alternative, GOSS (Gradient-based One-Side Sampling).

---

## Why Sample?

Sampling provides three benefits:

**Speed**: Fewer samples means faster histogram building. With 50% sampling, each tree trains roughly 2× faster.

**Regularization**: Each tree sees different data, reducing the risk of overfitting to specific examples.

**Variance reduction**: The ensemble averages over trees trained on different subsets, similar to bagging.

The tradeoff is potential accuracy loss. But smart sampling can minimize this.

---

## Random Subsampling

The simplest approach: randomly select a fraction of samples for each tree.

> [!tip] Algorithm: Random Subsampling
> ```
> Input: Dataset of n samples, subsample_rate
> 
> For each tree:
>     1. Select k = floor(n × subsample_rate) samples uniformly at random
>     2. Train tree on selected samples only
>     3. Use original gradients (no weight adjustment needed)
> ```

With `subsample = 0.5`, each tree trains on a random 50% of the data.

**Advantages**:
- Simple to implement
- Unbiased gradient estimates
- Good regularization

**Disadvantages**:
- May discard important samples
- Treats all samples equally, ignoring information about which are hard to predict

### When to Use

Random subsampling works well when:
- You want a simple regularization technique
- Dataset is large enough that missing some samples doesn't hurt
- Samples are roughly equally important

---

## The Insight: Not All Samples Are Equal

Here's the key observation: **samples with large gradients are more informative than samples with small gradients.**

Why? The gradient $g_i = \partial L / \partial F(x_i)$ tells us how much the loss would decrease if we adjusted the prediction for sample $i$. A large gradient means the model is making a big mistake on that sample. A small gradient means the model is already accurate there.

If we must subsample, we should keep the high-gradient samples (they drive learning) and subsample the low-gradient ones (they contribute less).

> [!example] Gradient Magnitudes
> In a classification problem with logistic loss, $g_i = \sigma(F) - y = p - y$:
> - **Hard positive**: $y = 1$, $p = 0.1$ → $g = -0.9$ (large negative gradient)
> - **Hard negative**: $y = 0$, $p = 0.9$ → $g = +0.9$ (large positive gradient)
> - **Easy sample**: $y = 1$, $p = 0.95$ → $g = -0.05$ (small gradient)
>
> Hard samples have 18× larger gradient magnitude than easy ones!

---

## GOSS: Gradient-based One-Side Sampling

GOSS (introduced in the LightGBM paper) exploits this insight. The name "One-Side" refers to keeping one side of the gradient distribution (the high-magnitude tail) intact while sampling the other side:

1. **Keep all samples with large gradients** (the "top" samples)
2. **Randomly subsample the rest** (the "other" samples)
3. **Upweight the subsampled samples** to maintain approximately unbiased estimates

> [!tip] Algorithm: GOSS Sampling
> ```
> Input: Gradients g[], top_rate, other_rate
>        e.g., top_rate = 0.2, other_rate = 0.1
> 
> 1. Compute importance scores: score[i] = |g[i]|
> 
> 2. Select top samples:
>    k_top = floor(n × top_rate)
>    top_indices = indices of k_top largest scores
> 
> 3. Sample from the rest:
>    other_indices = all indices not in top_indices
>    k_other = floor(len(other_indices) × other_rate)
>    sampled_other = random_sample(other_indices, k_other)
> 
> 4. Upweight the sampled-other samples:
>    weight = (1 - top_rate) / other_rate
>    For each i in sampled_other:
>        g[i] *= weight
>        h[i] *= weight
> 
> 5. Return top_indices + sampled_other
> ```

### Why Upweighting?

The "other" samples represent a larger population. If we sampled 10% of them, each sampled sample must "count for" 10 samples in the gradient sum.

The weight formula: $w = \frac{1 - \text{top\_rate}}{\text{other\_rate}}$

With `top_rate = 0.2` and `other_rate = 0.1`:
- The "other" population is 80% of data
- We sample 10% of that 80%
- Weight = $0.8 / 0.1 = 8$

This ensures the total gradient contribution from "other" samples remains correct in expectation. The estimate is approximately unbiased (the approximation improves with larger datasets).

> [!note] Multi-Class GOSS
> For multi-class classification, each sample has multiple gradients (one per class). The importance score becomes the sum of absolute gradients across classes: $\text{score}_i = \sum_k |g_{i,k}|$.

### Effective Sample Rate

How much data does GOSS actually use?

$$\text{effective\_rate} = \text{top\_rate} + \text{other\_rate} \times (1 - \text{top\_rate})$$

> [!example] Effective Sample Rate
> With `top_rate = 0.2` and `other_rate = 0.1`:
>
> $$\text{effective\_rate} = 0.2 + 0.1 \times 0.8 = 0.28$$
>
> GOSS uses 28% of the data per tree, yet keeps the most informative 20% intact.

Compare this to random 28% subsampling: GOSS gets the same speedup but with much less information loss.

---

## GOSS vs Random: Visual Comparison

```text
Dataset: 100 samples, sorted by gradient magnitude

Random Subsampling (28%):
| High gradient |  Medium gradient  | Low gradient |
| ●○●○●○●○●○    | ●○●○●○●○●○●○●○●○  | ○●○●○●○●○●   |
         (random selection across all samples)

GOSS (top=20%, other=10%):
| High gradient |  Medium gradient  | Low gradient |
| ●●●●●●●●●●    | ●○○○○○○○●○○○○○○○  | ○○○●○○○○○●   |
    (all top 20%)  (10% random from remaining 80%)

● = selected, ○ = not selected
```

GOSS guarantees all high-gradient samples are included, while random sampling may miss some.

---

## Column Sampling

In addition to row sampling, gradient boosting supports **column (feature) sampling**:

| Parameter | When Applied | Description |
|-----------|--------------|-------------|
| `colsample_bytree` | Once per tree | Sample features at the tree level |
| `colsample_bylevel` | Each depth level | Sample features per tree depth |
| `colsample_bynode` | Each split | Sample features per node |

The rates multiply together: $\text{effective} = \text{bytree} \times \text{bylevel} \times \text{bynode}$.

> [!example] Combined Column Sampling
> With `colsample_bytree = 0.8`, `colsample_bylevel = 0.8`, `colsample_bynode = 0.8`:
>
> At each node, you consider: $0.8 \times 0.8 \times 0.8 = 51.2\%$ of features.

Column sampling provides:
- **Regularization**: Prevents reliance on a few dominant features
- **Speed**: Fewer features to evaluate per split
- **Diversity**: Trees use different feature subsets

---

## Stochastic Gradient Boosting

When you combine row and column sampling, you get **Stochastic Gradient Boosting**:

```text
For each tree:
    rows = sample_rows(data, subsample_rate)    # or GOSS
    cols = sample_columns(features, colsample_bytree)
    
    For each node:
        node_cols = sample_columns(cols, colsample_bynode)
        histogram = build_histogram(rows, node_cols)
        split = find_best_split(histogram)
```

This introduces randomness at multiple levels:
- Which rows each tree sees
- Which features each tree/node considers
- Combined with learning rate shrinkage

The result is a diverse ensemble that is less prone to overfitting.

---

## GOSS Warm-Up Period

LightGBM disables GOSS for the first few iterations:

```text
warm_up_rounds = 1 / learning_rate

If iteration < warm_up_rounds:
    Use all data (no GOSS)
Else:
    Apply GOSS
```

**Why?** Early in training, the model is bad everywhere. Gradients don't yet reflect true sample difficulty. Once the model has learned basic patterns, gradients become meaningful indicators of which samples are genuinely hard.

With `learning_rate = 0.1`, this means about 10 warm-up rounds.

---

## When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Baseline | Random subsampling (`subsample = 0.8`) |
| Large dataset (> 1M rows) | GOSS for faster training |
| Imbalanced classification | GOSS (keeps minority class samples with high gradients) |
| Overfitting concerns | Lower `subsample`, add `colsample_bytree` |
| Small dataset (< 10k) | No sampling, or very mild (`subsample = 0.9`) |
| Debugging | Random (simpler to reason about) |

---

## Hyperparameters

### Row Sampling

| Parameter | Library | Default | Typical Range |
|-----------|---------|---------|---------------|
| `subsample` | Both | 1.0 | 0.5 - 1.0 |
| `boosting_type = 'goss'` | LightGBM | - | Enable GOSS |
| `top_rate` | LightGBM | 0.2 | 0.1 - 0.3 |
| `other_rate` | LightGBM | 0.1 | 0.05 - 0.2 |

### Column Sampling

| Parameter | Library | Default | Typical Range |
|-----------|---------|---------|---------------|
| `colsample_bytree` | Both | 1.0 | 0.5 - 1.0 |
| `colsample_bylevel` | XGBoost | 1.0 | 0.5 - 1.0 |
| `colsample_bynode` | Both | 1.0 | 0.5 - 1.0 |

> [!note] Starting Configuration
> For moderate regularization and speed:
> ```text
> subsample = 0.8
> colsample_bytree = 0.8
> colsample_bynode = 1.0
> ```
>
> For large datasets (> 1M rows) with LightGBM:
> ```text
> boosting_type = 'goss'
> top_rate = 0.2
> other_rate = 0.1
> ```

---

## Performance Impact

| Strategy | Effective Data | Speedup | Accuracy Trade-off |
|----------|----------------|---------|-------------------|
| Random 50% | 50% | ~2× | Moderate |
| Random 80% | 80% | ~1.25× | Small |
| GOSS (default) | 28% | ~3.5× | Small (keeps important samples) |
| + Column 80% | 28% × 80% features | ~4× | Moderate |

GOSS achieves better speedup-accuracy trade-offs than random sampling because it preserves the most informative samples.

> [!note] GOSS Memory Overhead
> GOSS requires storing sample weights, adding about 8 bytes per sample. For very large datasets, this is negligible compared to the histogram memory savings from using fewer samples.

---

## What's Next

We've covered how histogram-based training works and how sampling speeds it up. But what about categorical features and reducing the number of features?

The next post, [[gb-part-7|Exclusive Feature Bundling and Categorical Features]], explains LightGBM's EFB optimization (bundling sparse features) and how both libraries handle categorical data natively.

---

## Summary

**Sampling** speeds up training by using subsets of data:

- **Random subsampling**: Simple, uniform selection. Good baseline.
- **GOSS**: Keep high-gradient samples, subsample the rest. More efficient.

GOSS insight: samples with large gradients are more informative. Keeping them while subsampling easy samples preserves learning signal with less data.

**Column sampling** adds feature-level randomness, providing regularization and diversity.

| Method | Key Parameters | When to Use |
|--------|---------------|-------------|
| Random row | `subsample` | Baseline, small speedup needed |
| GOSS | `top_rate`, `other_rate` | Large datasets, need speed |
| Column | `colsample_bytree/bynode` | Regularization, feature diversity |

---

## References

1. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. Section 4: GOSS. [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

2. Friedman, J.H. (2002). "Stochastic Gradient Boosting". *Computational Statistics & Data Analysis*, 38(4), 367-378. [PDF](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf)

3. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)
