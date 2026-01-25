---
title: "Trees and the Split Gain Formula"
date: 2026-01-27
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - algorithms
  - decision-trees
description: "How decision trees fit into gradient boosting: from second-order Taylor expansion to the split gain formula that powers XGBoost and LightGBM."
aliases:
  - "gb-part-3"
  - "trees-and-split-gain"
series: "Inside Gradient Boosting"
series_order: 3
---

In [[gb-part-2|Part 2]], we saw that gradient boosting works by fitting weak learners to pseudo-residuals. At each round, we train a function $h$ that approximates the negative gradient of the loss.

But we left something unspecified: what kind of function is $h$? In principle, it could be anything. In practice, it's almost always a **decision tree**.

This post explains why trees dominate gradient boosting, then derives the formulas that make second-order optimization possible: the optimal leaf weight and the split gain formula.

---

## Why Trees?

Decision trees have properties that make them ideal weak learners:

**Non-linear without feature engineering**: A tree naturally captures interactions and non-linear patterns. You don't need to create polynomial features or specify interaction terms; the tree discovers them.

**No feature scaling required**: Trees only care about the *ordering* of values, not their magnitude. Features on different scales (age in years, income in dollars) work together without normalization.

**Handles missing values**: Modern tree implementations learn which direction to send missing values at each split. No imputation required.

**Constrained complexity**: By limiting depth or leaf count, we get a weak learner that captures broad patterns without overfitting. This is exactly what boosting needs.

**Fast to evaluate**: At inference time, a tree is just a sequence of comparisons. A depth-6 tree requires at most 6 feature lookups per prediction.

<!-- TODO: Visualization - A simple decision tree with 3 splits, showing how regions of feature space map to leaf predictions -->

---

## Anatomy of a Tree in Boosting

In gradient boosting, each tree is a function that maps inputs to corrections. Let's define the components:

**Splits**: Internal nodes that partition the data. A split tests a feature against a threshold: "Is `income < 50000`?" Samples go left if yes, right if no.

**Leaves**: Terminal nodes that output predictions. In standard GBDT, each leaf outputs a single constant value.

**Regions**: Each leaf corresponds to a region of feature space, defined by the conjunction of all split conditions on the path from root to that leaf.

The tree function can be written as:

$$
h(x) = \sum_{j=1}^{J} w_j \cdot \mathbf{1}[x \in R_j]
$$

where $J$ is the number of leaves, $w_j$ is the weight (prediction) of leaf $j$, $R_j$ is the region corresponding to leaf $j$, and $\mathbf{1}[x \in R_j]$ is 1 if $x$ falls in region $j$ and 0 otherwise.

> [!info] Trees as Basis Functions
> From an optimization perspective, each tree defines a set of "basis functions" (the indicator functions for each region). Training the tree selects and shapes these regions, while the leaf weights are the coefficients.

---

## The Second-Order Approximation

In [[gb-part-2|Part 2]], we used only the gradient (first derivative). XGBoost and LightGBM go further by using a **second-order Taylor expansion** of the loss.

Given a sample with true label $y$ and current prediction $F(x)$, we want to add a correction $h(x)$. The new prediction will be $F(x) + h(x)$.

We can approximate the loss after adding $h$ using a Taylor expansion:

$$L(y, F(x) + h(x)) \approx L(y, F(x)) + g \cdot h(x) + \frac{1}{2} H \cdot h(x)^2$$

where:
- $g = \frac{\partial L}{\partial F}$ is the gradient at the current prediction
- $H = \frac{\partial^2 L}{\partial F^2}$ is the Hessian (second derivative)

> [!info] Why "Second-Order"?
> The gradient tells us which direction decreases the loss. The Hessian tells us how quickly the loss *curves* in that direction. With both, we can estimate not just which way to go, but how far to step before we overshoot.

Summing over all samples and ignoring the constant $L(y, F(x))$ terms:

$$\tilde{\mathcal{L}}(h) = \sum_{i=1}^{n} \left[ g_i \cdot h(x_i) + \frac{1}{2} H_i \cdot h(x_i)^2 \right]$$

This is the objective we want to minimize when building tree $h$.

---

## Optimal Leaf Weights

Now, consider a specific tree structure. Each sample lands in exactly one leaf. For samples in leaf $j$, we can group them together:

$$\tilde{\mathcal{L}} = \sum_{j=1}^{J} \left[ G_j \cdot w_j + \frac{1}{2} H_j \cdot w_j^2 \right]$$

where:
- $G_j = \sum_{i \in I_j} g_i$ is the sum of gradients for samples in leaf $j$
- $H_j = \sum_{i \in I_j} H_i$ is the sum of Hessians for samples in leaf $j$
- $w_j$ is the leaf weight (what we're solving for)

This is a quadratic function in $w_j$, and it separates by leaf. For each leaf independently, we can take the derivative and set it to zero: $\frac{\partial \tilde{\mathcal{L}}}{\partial w_j} = G_j + H_j \cdot w_j = 0$. Solving gives us $w_j^* = -G_j/H_j$.

This is the **optimal leaf weight** when we have no regularization.

### Adding Regularization

To prevent overfitting, we add L2 regularization on leaf weights: $\Omega(h) = \frac{1}{2}\lambda \sum_{j=1}^{J} w_j^2$.

The objective becomes: $\tilde{\mathcal{L}} = \sum_{j=1}^{J} \left[ G_j \cdot w_j + \frac{1}{2} (H_j + \lambda) \cdot w_j^2 \right]$.

Taking the derivative and solving gives the **regularized optimal leaf weight**:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

The regularization term $\lambda$ appears in the denominator. Larger $\lambda$ shrinks leaf weights toward zero.

> [!example] Leaf Weight Calculation
> Suppose a leaf contains 3 samples with:
> - Gradients: $g_1 = 0.5$, $g_2 = -0.3$, $g_3 = 0.2$
> - Hessians: $H_1 = 1.0$, $H_2 = 0.8$, $H_3 = 1.2$
> - Regularization: $\lambda = 1.0$
>
> Sums: $G = 0.5 - 0.3 + 0.2 = 0.4$, $H = 1.0 + 0.8 + 1.2 = 3.0$
>
> Optimal weight: $w^* = -\frac{0.4}{3.0 + 1.0} = -0.1$
>
> The positive gradient sum means we're underpredicting; the negative weight will *increase* predictions (recall: the update is $F \leftarrow F + \eta \cdot h$, and our pseudo-residuals are negative gradients).

---

## The Split Gain Formula

Now we can answer the key question: **how do we decide where to split?**

Given a node with samples $I$, we're considering splitting it into left ($I_L$) and right ($I_R$) children. Is this split worthwhile?

### Objective Reduction from a Split

For the parent node alone (no split), the optimal objective value is: $\text{Obj}_\text{parent} = -\frac{1}{2} \frac{G^2}{H + \lambda}$. This comes from substituting the optimal weight $w^* = -G/(H+\lambda)$ back into the quadratic objective.

If we split, the two children have objectives: $\text{Obj}_\text{left} = -\frac{1}{2} \frac{G_L^2}{H_L + \lambda}$ and $\text{Obj}_\text{right} = -\frac{1}{2} \frac{G_R^2}{H_R + \lambda}$.

The **gain** from the split is the reduction in objective: $\text{Gain} = \text{Obj}_\text{parent} - \text{Obj}_\text{left} - \text{Obj}_\text{right}$.

Substituting and rearranging:

$$
\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right]
$$

<!-- TODO: Visualization - Bar chart comparing objective before split (one bar) vs after split (two bars), showing gain as the difference -->

### Adding Split Complexity Penalty

We often add a penalty $\gamma$ for each split to encourage simpler trees:

$$
\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma
$$

This is the **split gain formula** used by XGBoost and LightGBM.

> [!tip] The Split Gain Formula
> $$\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
>
> Where:
> - $G_L, H_L$ = gradient/Hessian sums for the left child
> - $G_R, H_R$ = gradient/Hessian sums for the right child
> - $\lambda$ = L2 regularization on leaf weights (`reg_lambda`)
> - $\gamma$ = minimum gain to make a split (`min_split_gain`, `gamma`)

> [!example] Split Gain Calculation
> Consider a node with 100 samples. We're evaluating whether to split on `age < 30`, which would send 40 samples left and 60 right. With $\lambda = 1$ and $\gamma = 0$:
>
> **Parent**: $G = 20$, $H = 100$
> - Score: $G^2/(H + \lambda) = 400/101 = 3.96$
>
> **Left child** (40 samples, age < 30): $G_L = 15$, $H_L = 40$
> - Score: $225/41 = 5.49$
>
> **Right child** (60 samples, age >= 30): $G_R = 5$, $H_R = 60$
> - Score: $25/61 = 0.41$
>
> **Gain**: $\frac{1}{2}(5.49 + 0.41 - 3.96) = 0.97$
>
> The split is worthwhile (gain > 0) because it separates high-gradient samples (left) from low-gradient samples (right). We'd compare this gain to all other candidate splits and pick the best.

---

## Understanding the Formula

Let's break down what the split gain formula tells us.

### The Score Terms

Each term $\frac{G^2}{H + \lambda}$ is the "score" of a node. It measures how much objective reduction we get from the optimal leaf weight.

**Intuition**: When gradients in a region all point the same direction (large $|G|$), we can make a strong correction. When they're mixed (small $|G|$), corrections cancel out. The score is higher when gradients are *aligned*.

### Why Squared Gradient?

The $G^2$ in the numerator might seem odd. Here's the intuition:

- If half the samples have $g_i = +1$ and half have $g_i = -1$, then $G = 0$
- No correction can help both groups; the optimal weight is zero
- Score is zero, reflecting that this node can't reduce the objective

- If all samples have $g_i = +1$, then $G = n$
- A strong negative correction helps everyone
- Score is large, reflecting potential for improvement

### The Role of Hessian

The Hessian $H$ in the denominator controls step size:

- Large $H$: loss curves sharply, so optimal weight is small (careful steps)
- Small $H$: loss is flat, so we can take larger steps

For squared error, $H_i = 1$ for all samples, so $H = n$ (sample count).

For logistic loss, $H_i = p_i(1 - p_i)$. Samples with confident predictions ($p \approx 0$ or $p \approx 1$) have small Hessians; uncertain samples ($p \approx 0.5$) have large Hessians.

### The Role of Regularization

**$\lambda$ (L2 regularization)**: Appears in every denominator. Larger $\lambda$ reduces the score difference between splits, preferring simpler trees with smaller weights.

**$\gamma$ (split penalty)**: Subtracts from the gain. If the structural improvement is less than $\gamma$, don't split. This is an explicit cost for tree complexity.

---

## Building a Tree: The Complete Picture

Now we can describe how a tree is built:

> [!tip] Algorithm: Tree Building with Second-Order Optimization
> ```
> Input: 
>   - Samples with gradients g[], Hessians H[]
>   - Parameters: max_depth, lambda, gamma, min_child_weight
> 
> 1. Create root node containing all samples
> 
> 2. While there are nodes to expand:
>     a. For each node:
>        - Compute G, H (gradient/Hessian sums)
>        
>     b. For each feature f:
>        - For each possible split threshold t:
>          - Compute G_L, H_L (samples with f < t)
>          - Compute G_R, H_R (samples with f >= t)
>          - Skip if H_L < min_child_weight or H_R < min_child_weight
>          - Compute Gain using the formula
>          - Track best (f, t, Gain)
>        
>     c. If best Gain > 0:
>        - Split node at (best_f, best_t)
>        - Create left and right children
>     d. Else:
>        - Mark as leaf
>        - Set weight w* = -G/(H + lambda)
> 
> 3. For all leaves: compute final weights
> 
> Output: Tree with splits and leaf weights
> ```

The expensive part is trying all features and thresholds (step 2b). With $n$ samples and $d$ features, naive enumeration is $O(n \cdot d)$ per split candidate, which quickly becomes prohibitive.

> [!warning] The Computational Bottleneck
> For a dataset with 1 million samples and 100 features, evaluating every possible split at every node is the dominant cost. [[gb-part-4|Part 4]] shows how histogram-based training reduces this from $O(n \log n)$ to $O(\text{bins})$ per feature.

---

## Regularization Hyperparameters

Let's connect the math to the hyperparameters you'll tune:

| Parameter | XGBoost | LightGBM | Effect |
|-----------|---------|----------|--------|
| $\lambda$ | `reg_lambda` | `lambda_l2` | Shrinks leaf weights toward zero |
| $\gamma$ | `gamma`, `min_split_loss` | `min_split_gain` | Minimum gain required to split |
| - | `min_child_weight` | `min_child_weight` | Minimum Hessian sum in leaf |
| - | `max_depth` | `max_depth` | Maximum tree depth |
| - | `max_leaves` | `num_leaves` | Maximum leaf count |

> [!note] Practical Defaults
> - Start with `lambda = 1.0`, `gamma = 0` (XGBoost defaults)
> - Increase `lambda` if overfitting on small datasets
> - Increase `gamma` to prune trees more aggressively
> - `min_child_weight` is the minimum Hessian sum allowed in a child. For squared error where $H_i = 1$, this equals minimum sample count. For classification where $H_i = p(1-p)$, samples with confident predictions contribute less, so 100 samples might only sum to 10 in Hessian.

---

## Special Case: Squared Error

For squared error loss, the formulas simplify nicely:

- Gradient: $g_i = F(x_i) - y_i$ (prediction minus target)
- Hessian: $H_i = 1$ (constant)

So $G = \sum_i (F(x_i) - y_i)$ and $H = n$ (sample count).

The optimal leaf weight becomes:

$$
w^* = -\frac{\sum_i (F(x_i) - y_i)}{n + \lambda} = \frac{\sum_i (y_i - F(x_i))}{n + \lambda}
$$

This is (approximately) the **mean residual** in the leaf, which matches our intuition from Part 1.

The split gain simplifies to comparing variance reduction, similar to classical CART.

---

## What's Next

We've derived the mathematics that power modern gradient boosting:

1. The second-order Taylor approximation of the loss
2. The optimal leaf weight formula
3. The split gain formula

But finding the best split still requires trying every feature and threshold. With millions of samples and hundreds of features, this is the computational bottleneck.

The next post, [[gb-part-4|Histogram-Based Split Finding]], shows how XGBoost and LightGBM solve this with histogram-based training, reducing complexity from O(n log n) to O(bins).

---

## Summary

**Second-order optimization** uses both gradient and Hessian to find better tree structures:

$$
w^* = -\frac{G}{H + \lambda}
$$

$$
\text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda} \right] - \gamma
$$

Key insights:

- **Squared gradients** measure alignment: how much can we correct together?
- **Hessians** control step size: how curved is the loss here?
- **$\lambda$** regularizes weights: shrink toward zero
- **$\gamma$** regularizes structure: don't split unless it's worth it
- **Trees are ideal weak learners**: non-linear, no scaling, missing values handled

---

## References

1. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)

2. Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine". *Annals of Statistics*, 29(5), 1189-1232. [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)

3. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
