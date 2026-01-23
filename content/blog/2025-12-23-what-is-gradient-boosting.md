---
title: "What is Gradient Boosting?"
date: 2025-12-23
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - algorithms
  - xgboost
  - lightgbm
description: "A deep dive into gradient boosting — from the additive ensemble model to functional gradient descent, the second-order approximation, and the innovations that power XGBoost and LightGBM."
series: "Gradient Boosting from Scratch"
series_order: 1
---

> [!abstract] Gradient Boosting from Scratch — Part 1 of 7
> This series explains gradient boosting internals, from theory to implementation.
>
> **In this post:** Functional gradient descent, second-order optimization, and the split gain formula.
> **Next:** [Histogram-Based Split Finding](/blog/2025-12-30-histogram-based-split-finding)

Gradient boosting is one of the most successful machine learning algorithms for tabular data. It dominates Kaggle competitions, powers fraud detection systems, and forms the backbone of many production ML pipelines. This post explains how it works from first principles—covering the theory in enough depth that you can read the original papers, while keeping the exposition accessible.

---

## The Additive Ensemble

Gradient boosting builds a prediction as a **sum of weak learners**:

$$
F_M(x) = \sum_{m=0}^{M} h_m(x)
$$

Here:
- $F_M(x)$ is our final prediction for input $x$ after $M$ boosting rounds
- Each $h_m(x)$ is a weak learner—typically a shallow decision tree
- $h_0(x)$ is usually a constant (e.g., the mean of targets for regression)

Individually, these trees are weak predictors. Combined, they form a powerful ensemble.

> [!question] Why shallow trees?
> Deep trees can fit training data almost perfectly on their own—but they overfit. By using shallow trees (depth 3–6) and combining many of them, we get a model that's both expressive and regularized. Each tree contributes a small correction.

But what does one of these trees actually look like? Consider this example:

![Decision tree diagram showing split nodes (blue) testing feature thresholds, and leaf nodes (green) outputting correction values.](/blog/assets/gradient-boosting/decision_tree.svg)

The **blue boxes** are *split nodes*—each tests a feature against a threshold. "Age < 30?" sends young samples left (Yes) and older samples right (No). The tree keeps splitting until it reaches the **green boxes**: *leaf nodes* that output a number.

Here's the key insight: in gradient boosting, these leaf values aren't class labels. They're *corrections*. Look at the values: $h = +1.2$, $h = +0.8$, $h = -0.3$. A young, high-income sample gets $+1.2$ added to its prediction. An older sample gets $+0.8$. These small corrections, accumulated across hundreds of trees, form the final prediction.

The key question: how do we choose what corrections each tree should output? The answer connects to gradient descent—but in function space rather than parameter space.

---

## The Loss Function

We want predictions that minimize a **loss function** over training data. 

For a single sample, we write $L(y, \hat{y})$ to denote the loss when the true value is $y$ and we predict $\hat{y}$. Common choices:

| Name | Formula | Use case |
|------|---------|----------|
| Squared error | $L(y, \hat{y}) = \tfrac{1}{2}(y - \hat{y})^2$ | Regression |
| Absolute error | $L(y, \hat{y}) = \lvert y - \hat{y} \rvert$ | Robust regression |
| Logistic | $L(y, \hat{y}) = \log(1 + e^{-y\hat{y}})$ | Binary classification |

The **total loss** over all $n$ training samples is:

$$
\mathcal{L} = \sum_{i=1}^{n} L\bigl(y_i, F(x_i)\bigr)
$$

We use the curly $\mathcal{L}$ for total loss to distinguish it from the per-sample loss $L$. Our goal is to find the prediction function $F$ that minimizes $\mathcal{L}$.

> [!note] Notation convention
> We write $F(x)$ when emphasizing the function, and $F(x_i)$ or just $F_i$ when referring to the prediction at a specific training point. Both refer to the same ensemble—just evaluated at different inputs.

---

## A Brief Detour: Parametric Gradient Descent

Before diving into gradient boosting, let's recall how gradient descent works in parametric models (like neural networks or linear regression).

In these models, we have parameters $\theta$ that we want to optimize. The update rule is:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

Let's break this down:
- $\theta$ represents all the model parameters (weights, biases, etc.)
- $\eta$ (eta) is the **learning rate**—a small positive number controlling step size
- $\nabla_\theta \mathcal{L}$ is the **gradient**—a vector of partial derivatives $\frac{\partial \mathcal{L}}{\partial \theta_j}$ for each parameter
- The gradient points in the direction of steepest *increase*, so we subtract it to *decrease* the loss

This works great when we have a fixed set of parameters. But what if our model is a sum of trees, where we keep adding new trees? This is where functional gradient descent comes in.

---

## Functional Gradient Descent

Gradient boosting takes a different approach: **gradient descent in function space**. Instead of updating parameters, we update the prediction function directly.

### The Functional Gradient

At each training point $x_i$, we can ask: "if we nudge our prediction $F(x_i)$ slightly, how does the loss change?" The answer is the partial derivative:

$$
\frac{\partial L\bigl(y_i, F(x_i)\bigr)}{\partial F(x_i)}
$$

This tells us the rate of change of the loss with respect to our prediction at that point.

> [!example] Squared error gradient
> For squared error $L(y, \hat{y}) = \tfrac{1}{2}(y - \hat{y})^2$:
> $$\frac{\partial L}{\partial F(x_i)} = \frac{\partial}{\partial F(x_i)}\left[\tfrac{1}{2}(y_i - F(x_i))^2\right] = -(y_i - F(x_i)) = F(x_i) - y_i$$
> The gradient is positive when we're over-predicting and negative when under-predicting.

### Pseudo-Residuals

We define the **pseudo-residual** at each sample as the *negative* gradient:

$$
r_i = -\frac{\partial L\bigl(y_i, F(x_i)\bigr)}{\partial F(x_i)}
$$

The term "pseudo-residual" comes from squared error loss, where this equals the actual residual $y_i - F(x_i)$. For other losses, it's different—but it always points in the direction that would reduce the loss.

> [!tip] Why "pseudo"?
> For losses other than squared error, the pseudo-residual isn't a literal residual. For logistic loss, it's related to how "surprised" the model is by the true label. But it serves the same purpose: telling the next tree which direction to push predictions.

---

## The Boosting Algorithm

The algorithm now follows naturally. At each round, we:
1. Compute pseudo-residuals (the direction to improve)
2. Fit a new tree to approximate those pseudo-residuals
3. Add the tree to our ensemble (taking a step in that direction)

> [!info] Gradient Boosting Algorithm
> **Initialize:** $F_0(x) = h_0$ (a constant, e.g., mean of $y_i$ for regression)
>
> **For** $m = 1, 2, \ldots, M$:
> 1. Compute pseudo-residuals:
>    $$r_i = -\frac{\partial L\bigl(y_i, F_{m-1}(x_i)\bigr)}{\partial F_{m-1}(x_i)} \quad \text{for } i = 1, \ldots, n$$
> 2. Fit weak learner $h_m$ to predict $(x_i, r_i)$
> 3. Update:
>    $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
>
> **Return:** $F_M(x)$

The parameter $\eta \in (0, 1]$ is the **learning rate** (also called **shrinkage**). Smaller values mean each tree contributes less, requiring more trees but typically improving generalization.

---

## Pseudo-Residuals for Common Losses

| Loss | $L(y, F)$ | Pseudo-residual $r = -\frac{\partial L}{\partial F}$ |
|------|-----------|------------------------------------------------------|
| Squared error | $\tfrac{1}{2}(y - F)^2$ | $y - F$ |
| Absolute error | $\lvert y - F \rvert$ | $\text{sign}(y - F)$ |
| Logistic | $\log(1 + e^{-yF})$ | $\dfrac{y}{1 + e^{yF}}$ |
| Poisson | $e^F - yF$ | $y - e^F$ |

> [!note] Classification labels
> For logistic loss, we use labels $y \in \{-1, +1\}$ rather than $\{0, 1\}$. This is a convention from the boosting literature that simplifies the math.

---

## Second-Order Optimization

The algorithm above uses only first-order gradients. XGBoost and LightGBM improve upon this by also using second-order information (the Hessian).

### Taylor Expansion

When we add a weak learner $h$ to the ensemble, we can approximate the new loss using a Taylor expansion around the current predictions:

$$
L(y, F + h) \approx L(y, F) + g \cdot h + \tfrac{1}{2} H \cdot h^2
$$

where:
- $g = \frac{\partial L}{\partial F}$ is the **gradient** (first derivative)
- $H = \frac{\partial^2 L}{\partial F^2}$ is the **Hessian** (second derivative)

Setting the derivative with respect to $h$ to zero gives the optimal step:

$$
h^* = -\frac{g}{H}
$$

This is a **Newton step**. It's smarter than a gradient step because it accounts for the curvature of the loss surface.

> [!example] Newton vs gradient descent
> Imagine minimizing $f(x) = x^2$. Gradient descent takes steps proportional to the slope: $\Delta x = -\eta \cdot 2x$. Newton's method divides by the curvature: $\Delta x = -\frac{2x}{2} = -x$, jumping directly to the minimum in one step. The Hessian tells us "how curved is the loss here?" and adjusts accordingly.

### Tree Building with Gradient and Hessian

For tree-based boosting, we aggregate gradients and Hessians within each leaf. Let $I_j$ denote the samples in leaf $j$:

$$
G_j = \sum_{i \in I_j} g_i, \qquad H_j = \sum_{i \in I_j} h_i
$$

The optimal prediction for leaf $j$ is:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

where $\lambda \geq 0$ is L2 regularization on leaf weights.

The **split gain**—the objective improvement from splitting a node—is:

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma
$$

Here $G_L, H_L$ are sums for the left child, $G_R, H_R$ for the right, and $\gamma$ penalizes adding leaves.

> [!tip] Reading split gain
> The gain formula compares the "before" (one node with $G, H$) to the "after" (two children). Higher gain means a more useful split. The regularization terms $\lambda$ and $\gamma$ prevent overfitting by penalizing complex trees.

To see why this formula works, look at what a split actually does:

![Split gain visualization: a parent node with mixed gradients (+/−) splits into children with grouped gradients.](/blog/assets/gradient-boosting/split_gain.svg)

The **parent node** (top) contains samples with *mixed* gradients—some positive (green circles, model under-predicting), some negative (red circles, over-predicting). When we split on a feature threshold, we're trying to *group* samples that need similar corrections.

After the split: the **left child** (green border) captures mostly positive gradients, while the **right child** (red border) captures mostly negative ones. Now look back at the formula: $G_L = \sum g_i$ in the left child is large and positive. $G_R$ in the right child is large and negative. When we square these sums ($G_L^2$ and $G_R^2$), we get large values.

Contrast this with the parent, where positive and negative gradients *cancel* when summed, making $G^2$ smaller. The gain formula captures exactly this: we want children where gradients *agree*, not where they fight each other.

### Gradients and Hessians for Common Objectives

| Objective | $g_i$ | $h_i$ |
|-----------|-------|-------|
| Squared error | $F(x_i) - y_i$ | $1$ |
| Logistic | $\sigma(F(x_i)) - y_i$ | $\sigma(F(x_i))(1 - \sigma(F(x_i)))$ |
| Softmax (class $k$) | $p_{ik} - \mathbf{1}_{y_i = k}$ | $p_{ik}(1 - p_{ik})$ |

Here $\sigma$ is the sigmoid function and $p_{ik}$ is the predicted probability for class $k$.

---

## Regularization

Gradient boosting is prone to overfitting. Common remedies:

> [!warning] Overfitting in gradient boosting
> Unlike random forests (which average independent trees), boosted trees are not independent—each corrects the previous. This means boosting can keep reducing training error indefinitely, even when test error starts increasing. Regularization is essential.

**Shrinkage ($\eta$)**: Small learning rates (0.01–0.1) with many trees. Each tree contributes less, preventing any single tree from overfitting.

**Subsampling**: Train each tree on a random fraction of rows or columns. This adds variance and reduces correlation between trees.

**Tree constraints**:
- `max_depth`: Limits tree depth (commonly 3–8)
- `min_child_weight`: Minimum Hessian sum required in a leaf
- `min_samples_leaf`: Minimum samples per leaf

**Objective regularization**: The $\lambda$ (L2 on weights) and $\gamma$ (leaf penalty) terms in the split gain formula.

---

## Historical Context

| Year | Development |
|------|-------------|
| 1997 | **AdaBoost** (Freund & Schapire): Boosting via adaptive sample reweighting |
| 1999 | **Gradient descent view** (Mason et al.): Connection to functional gradient descent |
| 2001 | **Gradient Boosting Machines** (Friedman): Shrinkage, subsampling, arbitrary losses |
| 2014 | **XGBoost** (Chen & Guestrin): Second-order optimization, regularized objective, systems engineering |
| 2017 | **LightGBM** (Ke et al.): Histogram-based splits, leaf-wise growth, GOSS |
| 2017 | **CatBoost** (Prokhorenkova et al.): Ordered boosting, native categorical handling |

> [!info] Why XGBoost was a breakthrough
> Friedman's 2001 algorithm was known in academia but wasn't widely used in practice. XGBoost (2014) combined second-order optimization with systems-level engineering: histogram-based splits, parallel training, cache-aware algorithms. This made gradient boosting practical for large-scale data and sparked the current era of GBDT dominance.

---

## Summary

Gradient boosting constructs an additive model by iteratively fitting weak learners to **pseudo-residuals**—the negative gradient of the loss:

$$
F_M(x) = \sum_{m=0}^{M} h_m(x)
$$

**Key concepts:**
- **Pseudo-residual**: $r_i = -\partial L / \partial F$ evaluated at each training point
- **Shrinkage**: Scale each tree's contribution by learning rate $\eta$
- **Second-order optimization**: Use gradient $g$ and Hessian $H$ for Newton-like updates
- **Split gain**: Guides tree construction using $G$ and $H$ sums

Modern implementations (XGBoost, LightGBM, CatBoost) add histogram-based split finding for efficiency—covered in the next post.

---

## References

1. Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine". *Annals of Statistics*, 29(5), 1189-1232. [JSTOR](https://www.jstor.org/stable/2699986)

2. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)

3. Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS 2017*. [Paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)

4. Mason, L. et al. (1999). "Boosting Algorithms as Gradient Descent in Function Space". *NeurIPS 1999*. [Paper](https://proceedings.neurips.cc/paper/1999/hash/96a93ba89a5b5c6c226e49b88973f46e-Abstract.html)

5. Schapire, R.E. & Freund, Y. (2012). *Boosting: Foundations and Algorithms*. MIT Press. (Comprehensive textbook on boosting)
