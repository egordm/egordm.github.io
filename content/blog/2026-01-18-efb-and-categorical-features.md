---
title: "Exclusive Feature Bundling and Categorical Features"
date: 2026-01-18
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - algorithms
  - optimization
description: "How LightGBM bundles sparse features to reduce memory 25×, and how both libraries handle categorical data without one-hot encoding."
aliases:
  - "gb-part-7"
  - "efb-and-categorical-features"
series: "Inside Gradient Boosting"
series_order: 7
---

> [!abstract] Inside Gradient Boosting, Part 7 of 9
> This series explains gradient boosting from first principles to advanced implementation details.
>
> **Previous:** [[gb-part-6|Gradient-Based Sampling (GOSS)]]
> **In this post:** Feature-level optimizations for sparse and categorical data.
> **Next:** [[gb-part-8|Regularization and Hyperparameter Tuning]]

So far, we've focused on sample-level optimizations: histograms, tree growth strategies, and sampling. But what about features?

Many real-world datasets have hundreds of sparse binary features (from one-hot encoding) or high-cardinality categorical columns. Both create computational challenges that specialized algorithms can address.

This post covers two related optimizations:
1. **Exclusive Feature Bundling (EFB)**: LightGBM's technique for combining sparse features
2. **Native categorical splits**: How to avoid one-hot encoding entirely

---

## The One-Hot Problem

When you one-hot encode a categorical feature, you create a sparse matrix:

```text
Original: color = {red, blue, green}

One-hot encoded:
  Sample 1 (color=red):   [1, 0, 0]
  Sample 2 (color=blue):  [0, 1, 0]
  Sample 3 (color=green): [0, 0, 1]
```

**Observation**: In any row, at most one of these features is non-zero. These features are **mutually exclusive** - meaning they cannot both be "on" at the same time.

The waste is significant:
- **3× memory**: Three columns instead of one
- **3× histogram work**: Build three separate histograms
- **3× split evaluation**: Test splits on three features

For a typical dataset with 50 categorical features averaging 20 categories each:
- One-hot encoding creates **1,000 features** instead of 50
- **20× overhead** in memory and computation

---

## Exclusive Feature Bundling (EFB)

LightGBM's insight: **mutually exclusive features can be merged into one without losing information.**

### The Core Idea

If features $f_1, f_2, f_3$ are mutually exclusive (at most one is non-zero per row), encode them as a single feature with distinct value ranges:

```text
Original features (each 0 or 1):
  f1: color_red
  f2: color_blue
  f3: color_green

Bundled feature:
  0 = all zero (missing/default)
  1 = f1 is active (red)
  2 = f2 is active (blue)
  3 = f3 is active (green)
```

One histogram instead of three. Same information, 3× less work.

### Finding Bundles

Not all features are perfectly exclusive. LightGBM uses a greedy algorithm (one that makes locally optimal choices at each step) to find near-exclusive bundles:

> [!tip] Algorithm: Exclusive Feature Bundling
> ```
> Input: Features, max_conflict_rate (default: 0.01%)
> 
> 1. Compute conflict graph:
>    - Edge between features i and j if both are non-zero in same row
> 
> 2. Sort features by non-zero count (descending)
> 
> 3. Greedy bundling:
>    For each feature:
>        Find bundle with lowest conflict
>        If conflict <= threshold: add to bundle
>        Else: create new bundle
> 
> Output: List of feature bundles
> ```

The conflict threshold (0.01% by default) allows bundling features that are *almost* exclusive, trading tiny accuracy loss for significant speedup.

### Encoding Bundle Values

Once features are bundled, encode the bundle value by assigning offset ranges:

```text
Bundle: [f1 (2 bins), f2 (3 bins), f3 (4 bins)]
Offsets: [0, 2, 5]

If f1 is active with bin 1: bundle_value = 0 + 1 = 1
If f2 is active with bin 2: bundle_value = 2 + 2 = 4
If f3 is active with bin 0: bundle_value = 5 + 0 = 5
```

The bundle has $2 + 3 + 4 = 9$ possible values, compactly represented in one column.

### Performance Impact

For a dataset with 105 one-hot features bundled into 14:

| Metric | Without EFB | With EFB |
|--------|-------------|----------|
| Features to process | 105 | 14 |
| Memory | 5 MB | 0.7 MB |
| Histogram work | 105 histograms | 14 histograms |

**Reduction**: ~7× faster, ~7× less memory.

---

## The Combinatorial Challenge of Categorical Splits

Beyond bundling, there's a deeper question: how do we split on categorical features natively?

For numerical features, we test thresholds: "is $x \le t$?"

For categorical features, we test set membership: "is $x \in S$?" where $S$ is a subset of categories.

The problem: for $k$ categories, there are $2^{k-1} - 1$ possible binary partitions.

| Categories | Possible Partitions |
|------------|---------------------|
| 3 | 3 |
| 5 | 15 |
| 10 | 511 |
| 20 | 524,287 |
| 100 | ~$10^{29}$ |

Exhaustive search is only feasible for small $k$.

---

## Native Categorical Splits

### Approach 1: One-Hot Strategy

For few categories (LightGBM default: $k \le 4$), test each category individually:

```text
For each category c:
    Compute gain for {c} vs {all other categories}
Return the best single-category split
```

This is $O(k)$ but only finds "isolate one category" splits, not arbitrary subsets.

### Approach 2: Gradient-Sorted Strategy

For more categories, LightGBM uses a clever approximation based on a classic result:

> [!info] Fisher's Theorem (1958)
> For squared error, the optimal binary partition of categories can be found by:
> 1. Sort categories by their mean target value
> 2. Test the $k-1$ split points in sorted order
> 
> The optimal split is contiguous in this ordering. For other losses (e.g., logistic), this is a heuristic that works well in practice.

For gradient boosting, we use gradient/Hessian ratio as the "mean target" proxy:

$$\text{ratio}_c = \frac{G_c}{H_c + \epsilon}$$

where $G_c$ and $H_c$ are gradient and Hessian sums for category $c$.

> [!tip] Algorithm: Gradient-Sorted Categorical Split
> ```
> Input: Categories, gradients, hessians
> 
> 1. For each category c:
>    ratio[c] = sum(gradients for c) / sum(hessians for c)
> 
> 2. Sort categories by ratio
> 
> 3. Scan sorted order:
>    Running sum: G_left, H_left = 0
>    For split point i = 1 to k-1:
>        Add category i-1 to left
>        Compute gain for left vs right
>        Track best split
> 
> Output: Categories going left, gain
> ```

**Complexity**: $O(k \log k)$ instead of $O(2^k)$.

Why does this work? Categories with similar gradient ratios have similar optimal leaf values and should be grouped together. Sorting by ratio orders categories by their "effect direction."

<!-- TODO: Visualization - Categories on a number line sorted by gradient ratio, showing the optimal split point -->

---

## Representing Categorical Splits

Once we find a good split, we need to store which categories go left.

### Bitset Storage

Store the "goes left" set as a bitset:

```text
Categories: {0: apple, 1: banana, 2: cherry, 3: date}
Split: {apple, cherry} go left

Bitset: 0b0101 = 5
        ||||
        |||+-- apple (0): 1 = left
        ||+--- banana (1): 0 = right
        |+---- cherry (2): 1 = left
        +----- date (3): 0 = right

Decision: if (bitset & (1 << category)) then LEFT else RIGHT
```

For up to 64 categories, this fits in a single `u64`. For more categories, LightGBM uses multiple 64-bit words packed together. Very high cardinality (1000+ categories) may require 16+ words per split, adding storage overhead.

---

## Preventing Overfitting

Categorical splits are prone to overfitting, especially with high cardinality. Each split can perfectly isolate small groups.

### LightGBM's Regularization Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cat_l2` | 10.0 | Extra L2 penalty on leaf weights |
| `cat_smooth` | 10.0 | Smoothing in ratio computation |
| `min_data_per_group` | 100 | Minimum samples per category |
| `max_cat_threshold` | 32 | Consider only top-k categories by importance |

> [!note] Practical Guidance
> For high-cardinality features (100+ categories):
> - Increase `cat_l2` to 20-50
> - Increase `min_data_per_group` to 200-500
> - Consider grouping rare categories into "other"

---

## When to Use Native vs One-Hot

### Use One-Hot Encoding When

- Cardinality is low (< 10 categories)
- Using a library without native support
- Interpretability matters (one feature per category)
- Categories have independent effects

### Use Native Categorical When

- Cardinality is moderate (10-1000 categories)
- Memory efficiency matters
- Categories should be grouped (similar products, locations)
- Using LightGBM or XGBoost with `enable_categorical`

### Consider Alternatives When

- Very high cardinality (> 1000): Target encoding, embeddings
- Ordinal relationship exists: Treat as numerical
- Too few samples per category: Group into "other"

---

## Handling Unknown Categories

At inference, you may encounter categories not seen during training.

| Strategy | Behavior | Use When |
|----------|----------|----------|
| **Default direction** | Go right (or learned default) | Production systems |
| **Error** | Reject the input | Safety-critical systems |
| **Map to "other"** | Reserve during training | Expected unknown categories |

> [!warning] Encoding Consistency
> Training and inference **must** use identical category-to-integer mappings. Store the encoder with your model artifact.

---

## Library Configuration

### LightGBM

```python
params = {
    'categorical_feature': [0, 3, 7],  # Column indices
    'max_cat_to_onehot': 4,            # Below this: one-hot search
    'cat_l2': 10.0,                    # Extra regularization
    'cat_smooth': 10.0,                # Ratio smoothing
    'max_cat_threshold': 32,           # Top-k categories
    'min_data_per_group': 100,         # Min samples
}
```

### XGBoost

```python
# Enable categorical support
dtrain = xgb.DMatrix(data, enable_categorical=True)

params = {
    'tree_method': 'hist',    # Required for categorical
    'max_cat_to_onehot': 4,   # One-hot threshold
}
```

---

## Summary: Feature-Level Optimizations

| Technique | Problem Solved | Speedup |
|-----------|----------------|---------|
| **EFB** | Sparse one-hot features | 5-25× memory, 5-10× speed |
| **Native categorical** | Arbitrary subset splits | Shallower trees, better accuracy |
| **Gradient-sorted splits** | Exponential split space | $O(k \log k)$ vs $O(2^k)$ |

### When EFB Kicks In

EFB is automatic in LightGBM. It activates when:
- Features are sparse (< 1% non-zero)
- Features are nearly exclusive (< 0.01% conflict)
- Multiple features can be bundled together

You don't need to configure anything; it just works.

---

## What's Next

We've covered the core algorithms: split finding, tree growth, sampling, and feature handling. The next post shifts to practical usage.

[[gb-part-8|Regularization and Hyperparameter Tuning]] explains how to configure all these parameters to get the best performance: learning rate, tree depth, regularization, and tuning strategies.

---

## References

1. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. Section 3.1: EFB. [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

2. Fisher, W.D. (1958). "On Grouping for Maximum Homogeneity". *Journal of the American Statistical Association*, 53(284), 789-798.

3. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)
