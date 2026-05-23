---
title: "Histogram-Based Split Finding"
date: 2026-01-12
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - algorithms
  - optimization
description: "The optimization that makes gradient boosting fast: how histogram-based training reduces split finding from O(n log n) to O(bins) with the subtraction trick."
aliases:
  - "gb-part-4"
  - "histogram-based-split-finding"
series: "Inside Gradient Boosting"
series_order: 4
---

In [[gb-part-3|Part 3]], we derived the split gain formula. To find the best split, we evaluate every feature and every possible threshold, then pick the one with highest gain.

But "every possible threshold" is the problem. With a million samples, each feature might have hundreds of thousands of unique values. Evaluating all of them is prohibitively slow.

This post explains **histogram-based training**, the optimization that makes modern gradient boosting practical. We'll see how discretizing features reduces complexity by orders of magnitude, and how the subtraction trick cuts histogram building in half.

---

## The Bottleneck: Exact Split Finding

Traditional (exact) split finding works like this:

1. For each feature, sort samples by that feature's value
2. Scan through sorted values, computing the split gain at each unique value
3. Track the best split across all features

The problem is step 1: sorting is $O(n \log n)$ per feature. With $d$ features, that's $O(d \cdot n \log n)$ per node. For a dataset with 1 million samples and 100 features, you're looking at billions of operations for each split decision.

> [!note] Why Sorting?
> The split gain formula needs $G_L$ and $H_L$ (gradient and Hessian sums for the left partition). By sorting samples and scanning left-to-right, we can compute these incrementally with running sums. Without sorting, we'd need to re-sum for every candidate threshold.

Early gradient boosting implementations (like the original GBM) used exact split finding, which limited them to small datasets.

---

## The Insight: Aggregate Statistics

Here's the key observation: **we only need gradient and Hessian sums, not individual values.**

The split gain formula involves $G_L = \sum_{i \in I_L} g_i$ and $H_L = \sum_{i \in I_L} h_i$. We don't care which specific samples are on each side; we only need the aggregate statistics.

This means we can:
1. Group samples into buckets (bins) based on feature values
2. Pre-compute gradient and Hessian sums for each bin
3. Evaluate splits at bin boundaries instead of individual values

If we use 256 bins instead of 1 million unique values, split finding becomes roughly 4000× faster (since 1,000,000 / 256 ≈ 4000).

---

## Feature Quantization

Before training, we convert continuous features to discrete bin indices. This is called **quantization** or **binning**.

### The Process

For each feature:
1. Determine bin boundaries from training data
2. Map each value to its corresponding bin index (0 to 255)
3. Store the quantized data as `u8` (1 byte per value per feature)

The quantized feature value is:

$$
\text{bin}(x) = \max\{j : b_j \le x\}
$$

where $b_0, b_1, \ldots, b_k$ are the bin boundaries.

### Quantile vs Uniform Binning

Two strategies for choosing bin boundaries:

**Uniform binning** divides the range into equal-width intervals:

```text
Values: [0, 1, 2, 3, 100]  bins=4
Boundaries: [0, 25, 50, 75, 100]
Result: [0, 0, 0, 0, 3]  ← All small values in one bin!
```

**Quantile binning** places boundaries so each bin has roughly equal samples:

```text
Values: [0, 1, 2, 3, 100]  bins=4
Boundaries: [0, 1, 2, 3, 100]
Result: [0, 1, 2, 3, 4]  ← Each value gets its own bin
```

> [!note] Why Quantile Binning Wins
> Quantile binning places bins where the data is, maximizing split resolution. XGBoost and LightGBM both default to quantile-based binning. XGBoost additionally uses **hessian-weighted** quantiles, giving more resolution to samples with higher curvature.

### Why 256 Bins?

Most implementations default to 256 bins because:
- **1 byte per value**: A `u8` can represent 0-255, minimizing memory
- **Sufficient resolution**: 256 possible split points per feature is enough for most problems
- **Cache-friendly**: Small histograms fit in CPU cache

With 256 bins, a dataset with 1M samples × 100 features shrinks from 400 MB (float32) to 100 MB (u8).

### Binning Hyperparameters

| Parameter | XGBoost | LightGBM | Default | Effect |
|-----------|---------|----------|---------|--------|
| Max bins | `max_bin` | `max_bin` | 256 | More bins = finer splits, more memory |
| Bin method | `tree_method='hist'` | (always histogram) | - | Enable histogram training |

> [!note] When to Change max_bin
> - **Increase** (512, 1024) for high-cardinality features where you see underfitting
> - **Decrease** (64, 128) for faster training or as regularization on small datasets
> - **Default 256** is a good starting point for almost all cases

---

## Building Histograms

A histogram stores gradient and Hessian sums for each bin:

```text
Feature histogram for feature f:
┌─────────────────────────────────────────────────┐
│ Bin 0: (sum_g = 1.5, sum_h = 0.8)              │
│ Bin 1: (sum_g = 2.3, sum_h = 1.2)              │
│ Bin 2: (sum_g = 0.1, sum_h = 0.3)              │
│ ...                                             │
│ Bin 255: (sum_g = 0.7, sum_h = 0.4)            │
└─────────────────────────────────────────────────┘
```

> [!tip] Algorithm: Build Histogram
> ```
> Input: 
>   - Quantized features X_bin[i, f] (bin index for sample i, feature f)
>   - Gradients g[i], Hessians h[i]
>   - Sample indices I belonging to this node
> 
> Output: Histogram H[f][b] = (sum_g, sum_h) for each feature f, bin b
> 
> Initialize: H[f][b] = (0, 0) for all f, b
> 
> For each sample i in I:
>     For each feature f:
>         b = X_bin[i, f]
>         H[f][b].sum_g += g[i]
>         H[f][b].sum_h += h[i]
> ```

The cost is $O(|I| \times d)$ where $|I|$ is the number of samples in the node and $d$ is the number of features.

---

## Split Finding from Histograms

Once the histogram is built, finding the best split for a feature is simple:

> [!tip] Algorithm: Find Best Split from Histogram
> ```
> Input: Histogram H[0..255] for one feature
> 
> 1. Compute totals:
>    G_total = sum(H[b].sum_g for all b)
>    H_total = sum(H[b].sum_h for all b)
> 
> 2. Scan left-to-right with running sums:
>    G_left = 0, H_left = 0
>    best_gain = 0, best_bin = -1
>    
>    For b = 0 to 254:   # 255 possible split points
>        G_left += H[b].sum_g
>        H_left += H[b].sum_h
>        G_right = G_total - G_left
>        H_right = H_total - H_left
>        
>        gain = 0.5 * (G_left²/(H_left+λ) + G_right²/(H_right+λ) - G_total²/(H_total+λ))
>        
>        If gain > best_gain:
>            best_gain = gain
>            best_bin = b
> 
> Output: best_bin, best_gain
> ```

The complexity is $O(\text{bins})$ per feature, or $O(\text{bins} \times d)$ for all features.

<!-- TODO: Visualization - Animation showing histogram scan with running sums accumulating left-to-right -->

---

## The Subtraction Trick

Here's where histogram-based training gets even faster.

When we split a node into left and right children, we need histograms for both. But we already have the parent's histogram. And histograms are additive: $\text{parent} = \text{left} + \text{right}$.

Therefore: $\text{right} = \text{parent} - \text{left}$.

**The subtraction trick**: Build the histogram for only one child, then subtract from the parent to get the other.

> [!example] Subtraction Trick in Action
> Consider splitting a parent node (1000 samples) into left (400 samples) and right (600 samples):
>
> **Step 1**: Build histogram for the *smaller* child (left, 400 samples). Cost: $O(400 \times d)$
>
> **Step 2**: Subtract from parent to get right histogram. Cost: $O(\text{bins} \times d)$
>
> **Without subtraction**: $O(400 \times d) + O(600 \times d) = O(1000 \times d)$
>
> **With subtraction**: $O(400 \times d) + O(256 \times d) \approx O(656 \times d)$
>
> **Savings**: ~35% in this case. On average across all splits, we save about 50%.

> [!note] Always Build the Smaller Child
> The subtraction trick is most effective when we build the histogram for the smaller child. On average, this saves about 50% of histogram building work.

---

## The Complete Picture

Let's trace through one split decision with histograms for a node with 10,000 samples, 100 features, and 256 bins.

**Step 1: Build histogram**

For each sample (10,000), for each feature (100), look up the bin index and add gradient/hessian to that bin. This costs approximately $10,000 \times 100 = 1,000,000$ operations. The result is 100 histograms, each with 256 entries containing $(g, h)$ pairs.

**Step 2: Find best split**

For each feature (100), scan through all bin boundaries (256) computing the split gain at each. This costs $100 \times 256 = 25,600$ operations.

**Step 3: Pick the best**

Compare gains across all features and pick the winner. Cost: 100 comparisons.

**Total**: ~1,000,000 operations, dominated by histogram building.

Compare this to exact split finding: sorting alone would cost $O(d \cdot n \log n) \approx 100 \times 10,000 \times 13 = 13,000,000$ operations.

---

## Complexity Comparison

| Operation | Exact | Histogram-Based |
|-----------|-------|-----------------|
| Sort/Partition | $O(n \log n)$ per feature | $O(1)$ (pre-quantized) |
| Build aggregates | $O(n)$ per feature | $O(n)$ per node, all features |
| Find split | $O(n)$ per feature | $O(\text{bins})$ per feature |
| **Total per node** | $O(d \cdot n \log n)$ | $O(d \cdot n)$ |
| **With subtraction** | - | $O(d \cdot n/2)$ average |

For typical values ($n = 10^6$, $d = 100$, $\text{bins} = 256$):
- Exact: ~2 billion operations per node
- Histogram: ~50 million operations per node
- **Speedup: ~40×**

---

## Missing Value Handling

Real data has missing values. How do histograms handle them?

**Strategy**: Track missing samples in a separate bin, then try both directions. For each candidate split, compute the gain with missing values going left, then with missing values going right, and choose whichever is better. Store the learned "default direction" with the split.

This adds minimal overhead (two scans instead of one) but gives the model the ability to learn the best handling for missing values, which often outperforms imputation.

---

## Parallel Histogram Building

Histogram building is the dominant cost, so parallelization matters.

### Feature-Parallel

Each thread handles different features:

```text
Thread 0: Build histogram for features 0-24
Thread 1: Build histogram for features 25-49
Thread 2: Build histogram for features 50-74
Thread 3: Build histogram for features 75-99

No synchronization needed - each thread writes to its own histograms.
```

This is the approach used by LightGBM and XGBoost for multi-core training.

### Data-Parallel

Each thread handles different samples, then results are merged:

```text
Thread 0: Partial histogram from samples 0-2499
Thread 1: Partial histogram from samples 2500-4999
Thread 2: Partial histogram from samples 5000-7499
Thread 3: Partial histogram from samples 7500-9999

Merge: Sum all partial histograms
```

This requires $O(\text{threads} \times \text{features} \times \text{bins})$ extra memory for partial histograms.

> [!note] Which is Better?
> Feature-parallel is usually preferred because it has no merge overhead and no extra memory. Data-parallel becomes useful in distributed settings where samples are partitioned across machines.

---

## What's Next

Histogram-based training answers "how do we evaluate splits fast?" But it doesn't answer "in what order should we split nodes?"

The next post, [[gb-part-5|Depth-Wise vs Leaf-Wise Tree Growth]], explores the two main strategies:
- **Depth-wise** (XGBoost default): Split all nodes at each level before moving deeper
- **Leaf-wise** (LightGBM default): Always split the highest-gain leaf

---

## Summary

**Histogram-based training** makes modern gradient boosting fast:

1. **Quantization**: Convert continuous features to discrete bins (typically 256)
2. **Histograms**: Aggregate gradient/Hessian sums per bin
3. **Fast split finding**: $O(\text{bins})$ instead of $O(n)$
4. **Subtraction trick**: Build only the smaller child's histogram, derive the other

Key insights:

- Split gain only needs **aggregate statistics**, not individual values
- 256 bins provide **sufficient resolution** while fitting in `u8`
- Subtraction trick saves **~50%** of histogram building work
- Feature-parallel building scales well on multi-core CPUs

$$
\text{Speedup} = \frac{O(n \log n)}{O(\text{bins})} \approx \frac{1,000,000 \times 20}{256} \approx 80,000\times
$$

(For split finding alone; total training speedup is typically 10-100×.)

---

## References

1. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)

2. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

3. Greenwald, M. & Khanna, S. (2001). "Space-Efficient Online Computation of Quantile Summaries". *SIGMOD 2001*. [PDF](http://infolab.stanford.edu/~datar/courses/cs361a/papers/quantiles.pdf)
