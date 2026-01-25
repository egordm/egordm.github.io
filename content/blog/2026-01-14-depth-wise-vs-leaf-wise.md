---
title: "Depth-Wise vs Leaf-Wise Tree Growth"
date: 2026-01-14
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - algorithms
  - decision-trees
description: "XGBoost grows trees level-by-level, LightGBM splits the best leaf. Learn when each strategy wins and how num_leaves relates to max_depth."
aliases:
  - "gb-part-5"
  - "depth-wise-vs-leaf-wise"
series: "Inside Gradient Boosting"
series_order: 5
---

In [[gb-part-4|Part 4]], we saw how histogram-based training makes split finding fast. But we left one question unanswered: **in what order should we split nodes?**

This turns out to matter a lot. Given a budget of $L$ leaves, different node orderings produce different trees with different accuracies. This post explores the two main strategies: **depth-wise** (XGBoost's default) and **leaf-wise** (LightGBM's default).

---

## The Question: Which Node to Split Next?

After the first split, we have two candidate nodes. We can:
1. Split both before going deeper (depth-wise)
2. Split whichever has higher potential gain (leaf-wise)

With a fixed leaf budget, these choices lead to very different trees:

```text
Depth-wise (balanced):        Leaf-wise (unbalanced):
       [root]                        [root]
        /  \                          /  \
      [A]  [B]                      [A]  [B]
      / \   / \                     / \    |
    [C][D][E][F]                  [C][D]  [B]
                                  / \
                                [G][H]
```

Which is better? It depends on your data, dataset size, and overfitting risk tolerance.

---

## Depth-Wise Growth

Depth-wise growth splits all nodes at each level before moving deeper. This is the traditional approach, used by XGBoost by default.

> [!tip] Algorithm: Depth-Wise Tree Growth
> ```
> Input: Training data, max_depth
> 
> 1. Create root node with all samples
> 
> 2. For depth = 0 to max_depth - 1:
>    a. For each leaf at current depth:
>       - Build histogram
>       - Find best split
>       - If gain > 0: create two children
>    b. If no splits made, stop early
> 
> 3. Assign leaf weights to all terminal nodes
> 
> Output: Balanced decision tree
> ```

**Key characteristic**: All leaves are at the same depth (or one level apart if some nodes stopped early).

### Advantages

**Balanced structure**: Every path from root to leaf has similar length. This makes the tree more interpretable and reduces the risk of overfitting to specific data paths.

**Easy parallelism**: All nodes at the same level are independent. You can build histograms and find splits for the entire level in parallel:

```text
Level 2: [C] [D] [E] [F]
         ↓   ↓   ↓   ↓   ← Process all four simultaneously
```

This level-synchronous approach is particularly valuable in distributed settings.

**Predictable memory**: At depth $d$, you have at most $2^d$ nodes. Memory usage is bounded and predictable.

### Disadvantages

**Wasted splits**: Depth-wise splits all nodes at a level, even low-gain ones. If node [E] has gain 0.1 while node [A] has gain 10.0, we still spend compute on [E].

**Less efficient for a fixed leaf budget**: With $L$ leaves, depth-wise may not allocate them optimally.

---

## Leaf-Wise Growth

Leaf-wise growth always splits the leaf with the highest potential gain, regardless of depth. This is LightGBM's default.

> [!tip] Algorithm: Leaf-Wise Tree Growth
> ```
> Input: Training data, num_leaves
> 
> 1. Create root node, add to priority queue
> 
> 2. While number of leaves < num_leaves:
>    a. Pop the leaf with highest gain from queue
>    b. Build histogram, find best split
>    c. If gain > 0:
>       - Split into two children
>       - Estimate potential gains for children
>       - Add both children to queue
>    d. Else: discard (can't split profitably)
> 
> 3. Assign leaf weights to all terminal nodes
> 
> Output: Possibly unbalanced decision tree
> ```

**Key characteristic**: Leaves can be at very different depths. The tree "grows toward" regions of high uncertainty.

> [!info] Priority Queue
> A priority queue keeps items sorted by a key (here, potential gain). "Pop" returns the item with the highest key. This ensures we always split the most promising leaf next.

### Advantages

**Optimal split allocation**: Given a budget of $L$ leaves, leaf-wise (greedily) picks the $L-1$ best splits. It never wastes a split on a low-gain node when a high-gain alternative exists.

**Faster training**: By focusing on high-gain nodes, leaf-wise often achieves the same loss reduction with fewer total splits.

**Better accuracy for fixed leaf count**: Empirically, leaf-wise trees with $L$ leaves often outperform depth-wise trees with $L$ leaves.

### Disadvantages

**Overfitting risk**: Leaf-wise can create very deep paths chasing outliers. A single unusual data point might cause the tree to grow 20 levels deep in one region.

**More complex implementation**: Requires a priority queue to track candidate leaves, and gains may need recomputation after sibling splits.

**Less parallelism-friendly**: The "best leaf" is a global decision, harder to distribute.

---

## Visual Comparison

Consider a dataset where one region has much more structure than another:

```text
Data: 8 samples
- 4 samples in Region A (high variance, needs more splits)
- 4 samples in Region B (low variance, one split suffices)

Depth-wise (max_depth=3, 8 leaves):
           [root]
          /      \
       [A:4]    [B:4]
       /   \    /   \
     [2]  [2] [2]  [2]   ← Equal splits in both regions
     /\   /\  /\   /\
    1 1  1 1 1 1  1 1

Leaf-wise (num_leaves=5):
           [root]
          /      \
       [A:4]    [B:4]     ← B stays as leaf (low gain)
       /   \       
     [2]  [2]             ← A gets all the splits
     /\   /\
    1 1  1 1
```

The numbers show sample counts. With the same leaf budget, leaf-wise concentrates splits where they reduce loss most.

---

## The num_leaves vs max_depth Relationship

These aren't equivalent parameterizations:

- A **full** tree of depth $d$ has $2^d$ leaves
- But depth-wise trees may have fewer leaves (if some splits are skipped)
- And leaf-wise trees may be deeper than $\log_2(\text{num\_leaves})$

> [!note] Rough Equivalence
> For similar model complexity:
> - `max_depth = 6` (depth-wise) ≈ `num_leaves = 63` (leaf-wise)
> - `max_depth = 8` ≈ `num_leaves = 255`
>
> But leaf-wise typically achieves better accuracy with the same leaf count.

This is why LightGBM defaults to `num_leaves = 31` rather than specifying depth.

---

## When to Use Which

| Criterion | Use Depth-Wise | Use Leaf-Wise |
|-----------|----------------|---------------|
| Dataset size | Small (< 10k) | Large (> 100k) |
| Overfitting risk | Higher concern | Lower concern |
| Tree depth needed | Shallow (≤ 6) | Deep |
| Hardware | GPU, distributed | Single-machine CPU |
| Priority | Stability | Speed + accuracy |

### Depth-Wise is Safer When...

**Small datasets**: With few samples, the risk of overfitting to noise is high. Depth-wise's balanced structure provides implicit regularization.

**Distributed training**: Level-synchronous computation maps naturally to distributed systems. XGBoost's distributed mode works best with depth-wise.

**GPU training**: Uniform tree structure allows more efficient GPU memory access patterns. When all trees have the same shape, kernel launches can be batched more effectively.

### Leaf-Wise Wins When...

**Large datasets**: With millions of samples, overfitting is less of a concern, and efficiency matters more.

**Training speed matters**: Leaf-wise typically converges faster for the same accuracy target.

**Deep trees are beneficial**: When your features have complex interactions requiring many splits, leaf-wise can build deep paths where needed.

---

## Hybrid: Leaf-Wise with max_depth

LightGBM lets you combine both constraints:

```text
# Leaf-wise with depth limit
num_leaves = 31      # Primary constraint
max_depth = 8        # Safety constraint (optional)
```

This gives you leaf-wise efficiency with depth-based regularization. If a leaf-wise path would go deeper than `max_depth`, that branch stops growing even if it has high gain.

> [!note] LightGBM Default
> By default, `max_depth = -1` (no limit). Set it explicitly if you're concerned about overfitting or want more interpretable trees.

### XGBoost's Loss-Guide Mode

XGBoost can also do leaf-wise:

```text
# Enable leaf-wise in XGBoost
grow_policy = 'lossguide'   # Instead of 'depthwise'
max_leaves = 31             # Leaf budget
```

This makes XGBoost behave more like LightGBM.

---

## Stopping Criteria

Both strategies use similar stopping conditions; they just differ in which splits are *prioritized*:

| Criterion | Description |
|-----------|-------------|
| `max_depth` | Don't grow beyond this depth |
| `num_leaves` / `max_leaves` | Maximum leaf count |
| `min_child_weight` | Minimum Hessian sum in child |
| `min_split_gain` | Minimum gain to accept split |
| `min_data_in_leaf` | Minimum samples per leaf |

These constraints work the same way regardless of growth strategy.

---

## Practical Recommendations

> [!note] Starting Configuration
> **For most problems, try leaf-wise first** (LightGBM defaults):
> - `num_leaves = 31` 
> - `max_depth = -1` (no limit) or `max_depth = 8` (for safety)
>
> Switch to depth-wise if:
> - You're seeing overfitting on a small dataset
> - You need distributed or GPU training
> - You want more uniform tree structures

### Tuning num_leaves

Start with `num_leaves = 31` and adjust:
- **Signs of underfitting**: training and validation loss both high, model too simple
- **Signs of overfitting**: training loss low but validation loss high, or validation loss increases

Adjustments:
- **Underfitting**: Increase `num_leaves` to 63, 127, 255
- **Overfitting**: Decrease `num_leaves` to 15, 7
- **Each doubling** roughly corresponds to adding one depth level

---

## What's Next

We've covered how trees are built and how split finding works. But building histograms over all samples is still expensive. Can we do better?

The next post, [[gb-part-6|Gradient-Based Sampling (GOSS)]], explains how LightGBM samples data intelligently: keeping high-gradient samples (where the model is wrong) and subsampling low-gradient ones.

---

## Summary

**Depth-wise growth** splits all nodes at each level before going deeper:
- Produces balanced trees
- Better for small datasets, distributed/GPU training
- XGBoost default (`grow_policy = 'depthwise'`)

**Leaf-wise growth** always splits the highest-gain leaf:
- Produces unbalanced trees optimized for loss reduction
- Better for large datasets, single-machine training
- LightGBM default

Key insight: for a fixed leaf budget, leaf-wise allocates splits more efficiently. But this efficiency comes with higher overfitting risk on small datasets.

| Strategy | Controls | Default In |
|----------|----------|------------|
| Depth-wise | `max_depth` | XGBoost |
| Leaf-wise | `num_leaves` | LightGBM |

---

## References

1. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

2. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)
