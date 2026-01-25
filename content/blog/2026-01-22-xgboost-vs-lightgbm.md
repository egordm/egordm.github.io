---
title: "XGBoost vs LightGBM: A Practical Comparison"
date: 2026-01-22
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - xgboost
  - lightgbm
description: "When to choose XGBoost vs LightGBM: speed, accuracy, features, and practical recommendations based on your use case."
aliases:
  - "gb-part-9"
  - "xgboost-vs-lightgbm"
series: "Inside Gradient Boosting"
series_order: 9
---

> [!abstract] Inside Gradient Boosting, Part 9 of 10
> This series explains gradient boosting from first principles to advanced implementation details.
>
> **Previous:** [[gb-part-8|Regularization and Hyperparameter Tuning]]
> **In this post:** A practical comparison of the two dominant implementations.
> **Next:** [[gb-part-10|Boosters: A From-Scratch Implementation in Rust]]

Throughout this series, I've referenced both XGBoost and LightGBM. Now it's time to compare them directly: when should you choose each, and what are the trade-offs?

This isn't a feature checklist. It's an analysis of **design philosophies** and **practical implications**.

---

## Design Philosophy

### XGBoost: Correctness and Compatibility

XGBoost prioritizes:
- **Reproducibility**: Same inputs → same outputs
- **Extensive documentation**: Every parameter is well-documented
- **Wide compatibility**: Multiple language bindings, serialization formats
- **Conservative defaults**: Less likely to overfit out of the box

### LightGBM: Speed and Efficiency

LightGBM prioritizes:
- **Training speed**: Optimized for large datasets
- **Memory efficiency**: Lower memory footprint
- **Aggressive optimizations**: GOSS, EFB, gradient quantization
- **Accuracy per compute**: Better results for the same training time

> [!note] Neither is "Better"
> They optimize for different objectives. The right choice depends on your constraints.

---

## Tree Growth Strategy

This is the core architectural difference:

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| **Default strategy** | Depth-wise | Leaf-wise |
| **Tree control** | `max_depth` | `num_leaves` |
| **Tree shape** | Balanced | Unbalanced |
| **Overfitting risk** | Lower | Higher (needs constraints) |

**Depth-wise** (XGBoost): Splits all nodes at each level before going deeper. Produces balanced trees.

**Leaf-wise** (LightGBM): Always splits the highest-gain leaf. Produces deeper, more efficient trees.

### Practical Impact

For the **same number of leaves**, leaf-wise typically achieves lower loss. But it can create very deep paths that overfit.

> [!note] Rough Equivalence
> `max_depth=6` (XGBoost) ≈ `num_leaves=63` (LightGBM)
>
> When comparing, set these to equivalent values.

---

## Speed Comparison

LightGBM is generally faster, especially on large datasets:

| Dataset Size | XGBoost | LightGBM | Typical Ratio |
|--------------|---------|----------|---------------|
| < 10K rows | Fast | Fast | ~1:1 |
| 100K rows | Moderate | Fast | 2-3× faster |
| 1M rows | Slow | Moderate | 3-5× faster |
| 10M+ rows | Very slow | Moderate | 5-10× faster |

*Note: Ratios are approximate and vary by dataset characteristics, feature count, and hyperparameters.*

### Why LightGBM is Faster

1. **Leaf-wise growth**: Skips low-gain splits
2. **GOSS**: Uses only 20-30% of data (high-gradient samples)
3. **EFB**: Bundles sparse features
4. **Histogram optimization**: Better cache utilization

### When XGBoost is Competitive

- Small datasets (< 100K rows): Overhead differences diminish
- GPU training: XGBoost's GPU implementation is highly optimized
- When GOSS/EFB don't apply (dense, non-sparse data)

---

## Accuracy Comparison

On most benchmarks, **accuracy is comparable** when both are properly tuned.

| Scenario | Edge |
|----------|------|
| Default settings | LightGBM (often) |
| Tuned settings | Tie |
| Small data (< 10K) | XGBoost (less overfit) |
| Large data (> 1M) | LightGBM (more efficient) |
| High-cardinality categoricals | LightGBM (native support) |

The lesson: **tuning matters more than library choice**.

---

## Feature Comparison

| Feature | XGBoost | LightGBM |
|---------|---------|----------|
| **Categorical handling** | Via `enable_categorical` | Native, gradient-sorted |
| **GOSS sampling** | ❌ | ✅ |
| **Feature bundling (EFB)** | ❌ | ✅ |
| **Leaf-wise growth** | Via `grow_policy='lossguide'` | Default |
| **GPU training** | ✅ (highly optimized) | ✅ |
| **Distributed training** | ✅ | ✅ |
| **Monotonic constraints** | ✅ | ✅ |
| **L1 regularization** | ✅ | ✅ |
| **Custom objectives** | ✅ | ✅ |
| **Model serialization** | JSON (documented schema) | Text/binary |

### Notable Differences

**Categorical features**: LightGBM's gradient-sorted algorithm finds better partitions than XGBoost's approach for high-cardinality features.

**Model format**: XGBoost's JSON format has a documented schema, making it easier for third-party tools to parse. LightGBM's format is simpler but less standardized.

**GPU performance**: XGBoost has a more mature GPU implementation, often faster on GPU than LightGBM.

---

## Parameter Mapping

When switching between libraries, use this mapping:

| XGBoost | LightGBM | Notes |
|---------|----------|-------|
| `eta` | `learning_rate` | Same meaning |
| `max_depth` | `max_depth` | Same, but LightGBM uses `num_leaves` by default |
| `gamma` | `min_split_gain` | Minimum gain for split |
| `min_child_weight` | `min_child_weight` | Minimum Hessian sum |
| `lambda` | `reg_lambda` | L2 regularization |
| `alpha` | `reg_alpha` | L1 regularization |
| `subsample` | `bagging_fraction` + `bagging_freq` | Row sampling |
| `colsample_bytree` | `feature_fraction` | Column sampling |
| `n_estimators` | `num_iterations` | Number of trees |

> [!warning] Default Differences
> Even with the same parameter names, defaults differ:
> - XGBoost `learning_rate` default: 0.3
> - LightGBM `learning_rate` default: 0.1
>
> Always set parameters explicitly for fair comparison.

---

## When to Choose XGBoost

✅ **Small to medium datasets** (< 100K rows)
✅ **GPU training** (XGBoost's GPU is very fast)
✅ **Reproducibility is critical** (deterministic behavior)
✅ **Model interoperability** (documented JSON format)
✅ **You want conservative defaults** (less likely to overfit)
✅ **Distributed training with Spark/Dask**

---

## When to Choose LightGBM

✅ **Large datasets** (> 100K rows)
✅ **Training speed is critical**
✅ **High-cardinality categorical features**
✅ **Sparse/one-hot encoded features** (EFB helps)
✅ **Memory constrained** (lower footprint)
✅ **Maximum accuracy per training time**

---

## Practical Recommendations

### Starting Point

For most problems, **start with LightGBM**:

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    num_leaves=31,
    # Enable early stopping
)
model.fit(X_train, y_train,
          eval_set=[(X_valid, y_valid)],
          callbacks=[lgb.early_stopping(50)])
```

### When to Switch

Consider XGBoost if:
- LightGBM is overfitting on small data
- You need GPU training
- Model serialization/interoperability matters
- You want more predictable, documented behavior

### Both

For production or competitions, **try both and compare**:

```python
# Run both with equivalent settings
# Compare validation performance
# Choose winner
```

The best library is the one that works best for your specific data.

---

## Summary

| Dimension | XGBoost | LightGBM |
|-----------|---------|----------|
| **Speed** | Good | Better |
| **Memory** | Higher | Lower |
| **Defaults** | Conservative | Aggressive |
| **Categoricals** | Basic | Advanced |
| **Documentation** | Excellent | Good |
| **GPU** | Excellent | Good |
| **Serialization** | JSON (standard) | Text (simple) |

### The Bottom Line

- **LightGBM**: Default choice for speed and efficiency
- **XGBoost**: When you need compatibility, GPU, or conservative behavior

Both are excellent. The 5% accuracy difference you might be worrying about is usually recovered by better hyperparameter tuning, not by switching libraries.

---

## What's Next

We've covered the theory, the algorithms, and the practical usage. In the final post, I'll share my own journey: building a gradient boosting library from scratch in Rust.

[[gb-part-10|Boosters: A From-Scratch Implementation in Rust]] walks through the design decisions, performance optimizations, and lessons learned from implementing everything we've discussed in this series.

---

## References

1. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)

2. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

3. [XGBoost Documentation](https://xgboost.readthedocs.io/)

4. [LightGBM Documentation](https://lightgbm.readthedocs.io/)
