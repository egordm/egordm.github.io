---
title: "Regularization and Hyperparameter Tuning"
date: 2026-02-01
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - hyperparameters
  - optimization
description: "A practical guide to configuring gradient boosting: learning rate, tree depth, regularization, and systematic tuning strategies."
aliases:
  - "gb-part-8"
  - "regularization"
series: "Inside Gradient Boosting"
series_order: 8
---

> [!abstract] Inside Gradient Boosting, Part 8 of 10
> This series explains gradient boosting from first principles to advanced implementation details.
>
> **Previous:** [[gb-part-7|EFB and Categorical Features]]
> **In this post:** How to configure and tune gradient boosting for optimal performance.
> **Next:** [[gb-part-9|XGBoost vs LightGBM: A Practical Comparison]]

We've covered the algorithms. Now for the practical question: **how do you configure all these parameters?**

Gradient boosting has many hyperparameters, each affecting model behavior differently. This post provides a systematic approach to tuning, organized by what each parameter controls.

---

## The Core Trade-Off: Bias vs Variance

Every hyperparameter affects the bias-variance trade-off:

- **Bias** (underfitting): Model is too simple to capture patterns
- **Variance** (overfitting): Model memorizes training data, fails on new data

Gradient boosting defaults are often biased toward overfitting (deep trees, no sampling). Tuning is mostly about adding the right amount of regularization.

---

## Learning Rate and Number of Trees

These two parameters work together as the most important knobs.

### learning_rate (eta)

Controls how much each tree contributes: $F_m = F_{m-1} + \eta \cdot h_m$.

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.3 (default) | Aggressive | Quick experiments |
| 0.1 | Balanced | Good starting point |
| 0.01-0.05 | Conservative | Maximum accuracy |

Lower learning rates need more trees but typically generalize better.

### n_estimators

Number of boosting rounds (trees).

> [!note] The Golden Rule
> **Set a high `n_estimators` and use early stopping.** Let the validation loss determine when to stop.
>
> ```python
> model.fit(X_train, y_train,
>           eval_set=[(X_valid, y_valid)],
>           early_stopping_rounds=50)
> ```

Early stopping finds the optimal number of trees automatically, avoiding both underfitting and overfitting.

### The Relationship

Lower learning rate + more trees = better generalization (given enough trees).

| Configuration | Typical n_estimators | Notes |
|---------------|----------------------|-------|
| `learning_rate=0.3` | 100-300 | Fast, may overfit |
| `learning_rate=0.1` | 300-1000 | Balanced |
| `learning_rate=0.01` | 1000-5000 | Best accuracy, slow |

---

## Tree Structure Parameters

These control how complex each individual tree can be.

### max_depth

Maximum levels in each tree.

| Value | Effect |
|-------|--------|
| 3-4 | Simple interactions (linear + pairwise) |
| 6 (default) | Moderate complexity |
| 8-10 | Complex interactions |
| > 10 | Usually overkill, high overfit risk |

**Rule of thumb**: Start with 6, reduce if overfitting, increase if underfitting.

### num_leaves (LightGBM)

Alternative to max_depth for leaf-wise growth. Maximum leaves per tree.

| num_leaves | Rough max_depth equivalent |
|------------|---------------------------|
| 31 (default) | ~5 |
| 63 | ~6 |
| 127 | ~7 |
| 255 | ~8 |

### min_child_weight / min_data_in_leaf

Minimum sum of Hessians (or minimum samples) in a leaf.

- Higher values = more conservative, stops early
- Lower values = trees can create smaller leaves
- **Typical range**: 1-100

> [!note] Hessian vs Samples
> For squared error, Hessian = 1 per sample, so `min_child_weight` ≈ minimum samples.
> For logistic loss, Hessian = $p(1-p)$, which peaks at 0.25 for uncertain predictions.

### min_split_gain (gamma)

Minimum gain required to make a split. Pruning parameter.

$$\text{Split if: } \text{Gain} > \gamma$$

- 0 (default): Allow all positive-gain splits
- 0.1-1.0: Mild pruning
- 1-5: Aggressive pruning

---

## Regularization Parameters

Beyond tree structure, explicit regularization on leaf weights.

### reg_lambda (L2)

L2 regularization on leaf weights. Appears in the optimal weight formula:

$$w^* = -\frac{G}{H + \lambda}$$

| Value | Effect |
|-------|--------|
| 0 | No regularization |
| 1 (default) | Mild |
| 5-10 | Strong |

Higher $\lambda$ shrinks leaf weights toward zero, reducing variance.

### reg_alpha (L1)

L1 regularization promotes sparse leaf weights (some leaves become exactly zero).

- Rarely needed for GBDT
- Useful when you want interpretable sparse models
- **Typical range**: 0-1

---

## Sampling Parameters

Randomness as regularization.

### subsample (row sampling)

Fraction of samples used per tree.

| Value | Effect |
|-------|--------|
| 1.0 (default) | Use all data |
| 0.8 | 20% dropout, mild regularization |
| 0.5 | 50% dropout, strong regularization |

Lower values = faster training, more regularization, but may hurt accuracy.

### colsample_bytree / colsample_bynode

Feature sampling per tree or per node.

| Value | Effect |
|-------|--------|
| 1.0 (default) | Use all features |
| 0.8 | 20% feature dropout |
| 0.5 | Random forest-like feature selection |

Feature sampling is especially useful when you have many correlated features.

---

## Putting It All Together: Tuning Strategy

### Step 1: Baseline

Start with sensible defaults:

```python
params = {
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 6,           # or num_leaves=31
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'reg_lambda': 1.0,
    'early_stopping_rounds': 50,
}
```

### Step 2: Diagnose

Compare training and validation loss curves:

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Both losses high | Underfitting | Increase complexity (depth, trees) |
| Training low, validation high | Overfitting | Add regularization |
| Validation loss increases | Overfitting | Reduce iterations (early stopping) |
| Losses close, both decreasing | Good fit | Continue or slight regularization |

### Step 3: Tune Tree Structure

If overfitting:
- Reduce `max_depth` (6 → 4)
- Increase `min_child_weight` (1 → 10)
- Add `min_split_gain` (0 → 0.1)

If underfitting:
- Increase `max_depth` (6 → 8)
- Decrease `min_child_weight`
- Lower `learning_rate`, increase `n_estimators`

### Step 4: Add Sampling

Add subsample and colsample for regularization:

```python
params['subsample'] = 0.8
params['colsample_bytree'] = 0.8
```

Often helps generalization with minimal accuracy cost.

### Step 5: Fine-Tune with Cross-Validation

Use systematic search for the final push:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Or use Optuna, Hyperopt, etc. for Bayesian optimization
```

---

## Quick Reference: Parameter Effects

| Parameter | Increase → | Decrease → |
|-----------|-----------|------------|
| `learning_rate` | Faster convergence, overfit | Slower, better generalization |
| `n_estimators` | More capacity, overfit risk | Less capacity, underfit |
| `max_depth` | Complex trees, overfit | Simple trees, underfit |
| `min_child_weight` | Conservative, underfit | Aggressive, overfit |
| `reg_lambda` | Shrink weights, underfit | Free weights, overfit |
| `subsample` | Less regularization | More regularization |
| `colsample_bytree` | Less regularization | More regularization |

---

## Library-Specific Defaults

| Parameter | XGBoost Default | LightGBM Default |
|-----------|-----------------|------------------|
| `learning_rate` | 0.3 | 0.1 |
| `max_depth` | 6 | -1 (unlimited) |
| `num_leaves` | - | 31 |
| `min_child_weight` | 1 | 20 |
| `reg_lambda` | 1 | 0 |
| `subsample` | 1 | 1 |

Note the different philosophies: XGBoost restricts by depth, LightGBM restricts by leaves.

---

## Common Mistakes

> [!warning] Mistakes to Avoid
> 1. **Tuning `n_estimators` directly** — Use early stopping instead
> 2. **Ignoring validation curves** — Always plot training vs validation loss
> 3. **Tuning too many parameters at once** — Follow the staged approach
> 4. **Using defaults for production** — Defaults are for quick exploration
> 5. **Forgetting to scale learning rate** — When reducing learning rate, increase iterations

---

## What's Next

Now you know how to configure gradient boosting. But should you use XGBoost or LightGBM?

The next post, [[gb-part-9|XGBoost vs LightGBM: A Practical Comparison]], compares the two libraries across dimensions that matter: speed, accuracy, features, and when to choose each.

---

## Summary

**Core parameters** (tune first):
- `learning_rate` + `n_estimators` with early stopping
- `max_depth` or `num_leaves`

**Regularization** (add if overfitting):
- `min_child_weight`, `min_split_gain`
- `reg_lambda`, `reg_alpha`
- `subsample`, `colsample_bytree`

**Tuning process**:
1. Start with defaults + early stopping
2. Diagnose: compare training vs validation loss
3. Adjust tree structure
4. Add sampling regularization
5. Fine-tune with cross-validation

The magic formula: **lower learning rate + more trees + early stopping** = robust generalization.

---

## References

1. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)

2. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

3. Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine". *Annals of Statistics*. [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
