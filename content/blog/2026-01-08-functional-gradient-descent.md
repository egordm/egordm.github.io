---
title: "Functional Gradient Descent"
date: 2026-01-08
draft: false
tags:
  - machine-learning
  - gradient-boosting
  - algorithms
  - optimization
description: "How gradient boosting optimizes in function space: from pseudo-residuals to the complete boosting algorithm, with derivations for common loss functions."
aliases:
  - "gb-part-2"
  - "functional-gradient-descent"
series: "Inside Gradient Boosting"
series_order: 2
---

In [[gb-part-1|Part 1]], we built the intuition: gradient boosting trains weak learners sequentially, each one correcting the errors of the previous ensemble. We saw that for squared error, the "correction" is just the residual.

But where does the "gradient" in gradient boosting come from? Why does training on residuals work? And how do we extend this to classification, ranking, and other tasks?

This post answers those questions by formalizing gradient boosting as **optimization in function space**. We'll derive the pseudo-residual, work through concrete loss functions, and see why this framework is so powerful.

---

## From Parameter Space to Function Space

Let's start with what you probably already know: gradient descent.

> [!info] Gradient Descent Refresher
> Given parameters $\theta$ and a loss function $\mathcal{L}(\theta)$, gradient descent updates:
> 
> $$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$
> 
> We compute the gradient (the direction of steepest ascent) and move in the opposite direction (toward lower loss). The learning rate $\eta$ controls the step size.

This works when your model has a fixed structure and you're just tuning its parameters. A neural network, for instance, has weights $\theta$ that gradient descent adjusts.

But in gradient boosting, we're not adjusting parameters of a single model. We're **adding entirely new models** to an ensemble. The "thing we're optimizing" is the prediction function itself:

$$
F(x) = f_1(x) + f_2(x) + f_3(x) + \cdots + f_M(x)
$$

How do you do gradient descent when your variable is a *function*?

---

## The Functional View

Here's the key insight: at each training point $x_i$, the prediction $F(x_i)$ is just a number. We can think of the function $F$ as a vector of length $n$ (one component per training sample):

$$\mathbf{F} = \begin{bmatrix} F(x_1) \\ F(x_2) \\ \vdots \\ F(x_n) \end{bmatrix}$$

Now our loss becomes a function of this vector: $\mathcal{L}(\mathbf{F}) = \sum_{i=1}^{n} L(y_i, F(x_i))$, where $L(y, F)$ is the per-sample loss (e.g., squared error, logistic loss).

We can take the gradient of this loss with respect to the *predictions*:

$$\nabla_\mathbf{F} \mathcal{L} = \begin{bmatrix} \frac{\partial L}{\partial F(x_1)} \\ \frac{\partial L}{\partial F(x_2)} \\ \vdots \\ \frac{\partial L}{\partial F(x_n)} \end{bmatrix}$$

This is the direction in prediction space that increases the loss fastest. To reduce the loss, we want to move in the opposite direction: the **negative gradient**.

<!-- TODO: Visualization - 2D surface showing loss as function of F(x₁) and F(x₂), with gradient vector and descent direction -->

---

## The Pseudo-Residual

Here's where the magic happens. We want to update our function: $F \leftarrow F - \eta \cdot \nabla_F \mathcal{L}$.

But we can't just add arbitrary vectors to $F$. We need to add a *function* that generalizes to new data. So we train a weak learner $h$ to approximate the negative gradient: $h \approx -\nabla_F \mathcal{L}$.

This $h$ is trained to predict the **negative gradient at each training point**. The target for the weak learner at sample $i$ is:

$$
r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}
$$

This target is called the **pseudo-residual**. For regression losses, it's closely related to the actual residual; for other losses, it's more general.

> [!info] Why "Pseudo-Residual"?
> For squared error loss, the pseudo-residual equals the residual: $r_i = y_i - F(x_i)$. For other losses, the pseudo-residual is the negative gradient, which points toward improvement but isn't exactly the "error" in the intuitive sense. Friedman (2001) coined the term to emphasize this generality.

Once we've trained $h$ to fit these pseudo-residuals, we update: $F_{\text{new}} = F_{\text{old}} + \eta \cdot h$.

And repeat. Each iteration adds a function that pushes predictions in the direction of lower loss.

---

## Deriving Pseudo-Residuals for Common Losses

Let's work through the math for the most common loss functions.

### Squared Error (Regression)

The loss for a single sample is $L(y, F) = \frac{1}{2}(y - F)^2$. (The $\frac{1}{2}$ is for convenience; it cancels when we differentiate.)

Taking the derivative: $\frac{\partial L}{\partial F} = -(y - F) = F - y$.

The **negative** gradient (our pseudo-residual) is:

$$r = -\frac{\partial L}{\partial F} = y - F$$

This is exactly the residual. Training on residuals *is* functional gradient descent for squared error.

> [!example] Squared Error Pseudo-Residual
> If $y = 100$ and $F(x) = 80$:
> - Gradient: $\frac{\partial L}{\partial F} = 80 - 100 = -20$ (loss decreases if we increase prediction)
> - Pseudo-residual: $r = -(-20) = +20$ (nudge prediction up by 20)

---

### Logistic Loss (Binary Classification)

For binary classification with labels $y \in \{0, 1\}$, the log-loss is: $L(y, F) = -y \cdot \log(\sigma(F)) - (1-y) \cdot \log(1 - \sigma(F))$, where $\sigma(F) = \frac{1}{1 + e^{-F}}$ is the sigmoid function and $F$ is the log-odds.

This simplifies to $L(y, F) = \log(1 + e^{-F}) + (1-y) \cdot F$.

Taking the derivative and simplifying: $\frac{\partial L}{\partial F} = \sigma(F) - y$.

The pseudo-residual is:

$$r = y - \sigma(F) = y - p$$

where $p = \sigma(F)$ is the predicted probability.

> [!example] Logistic Loss Pseudo-Residual
> If $y = 1$ (positive class) and $p = 0.3$ (model predicts 30% probability):
> - Pseudo-residual: $r = 1 - 0.3 = +0.7$
> - Interpretation: the model should increase its raw score for this sample.
>
> If $y = 0$ (negative class) and $p = 0.8$:
> - Pseudo-residual: $r = 0 - 0.8 = -0.8$
> - Interpretation: the model should decrease its raw score for this sample.

---

### Absolute Error (Robust Regression)

For absolute error (L1 loss): $L(y, F) = |y - F|$.

The derivative is: $\frac{\partial L}{\partial F} = -\text{sign}(y - F)$.

The pseudo-residual is: $r = \text{sign}(y - F)$.

This is just the sign of the residual: either $+1$ or $-1$. The magnitude doesn't matter; absolute error only cares about the direction of the error, making it robust to outliers.

---

### Summary of Common Losses

| Loss | Formula | Pseudo-Residual |
|------|---------|-----------------|
| Squared Error | $\frac{1}{2}(y - F)^2$ | $y - F$ |
| Absolute Error | $\|y - F\|$ | $\text{sign}(y - F)$ |
| Logistic (binary) | $-y\log(p) - (1-y)\log(1-p)$ | $y - p$ |

The pattern: compute $-\partial L / \partial F$, that's your training target.

---

## The Complete Boosting Algorithm

Now we can write down gradient boosting in its general form:

> [!tip] Algorithm: Gradient Boosting (General)
> ```
> Input:
>   - Training data (X, y) with n samples
>   - Differentiable loss function L(y, F)
>   - Number of rounds M
>   - Learning rate η
> 
> Initialize: F₀(x) = argmin_c Σᵢ L(yᵢ, c)
>             (e.g., mean for squared error, log-odds for logistic)
> 
> For m = 1 to M:
>     1. Compute pseudo-residuals:
>        rᵢ = -∂L(yᵢ, F_{m-1}(xᵢ)) / ∂F_{m-1}(xᵢ)
>     
>     2. Fit weak learner hₘ to (X, r):
>        hₘ = argmin_h Σᵢ (rᵢ - h(xᵢ))²
>     
>     3. Update ensemble:
>        Fₘ = F_{m-1} + η · hₘ
> 
> Output: F_M
> ```

A few notes on this algorithm:

**Initialization**: We start with the constant that minimizes the loss over all training samples. For squared error, this is the mean of $y$. For logistic loss, it's the log-odds of the positive class.

> [!example] Initialization for Classification
> If your training data has 70% positive samples ($\bar{y} = 0.7$):
> $$F_0 = \log\left(\frac{0.7}{1 - 0.7}\right) = \log\left(\frac{0.7}{0.3}\right) \approx 0.85$$
> This gives an initial probability of $\sigma(0.85) \approx 0.70$, matching the class balance.

**Fitting the weak learner**: We always fit to the pseudo-residuals using squared error, regardless of the original loss function. This is because we're approximating the gradient direction, not directly optimizing the original loss.

**Why squared error for the weak learner?** The negative gradient is a continuous value at each point. Fitting a tree to these values with squared error finds a function that points (approximately) in the gradient direction. This works even when the original loss is not squared error.

---

## Why Shrinkage Works

In [[gb-part-1|Part 1]], we saw that shrinkage (small $\eta$) improves generalization. Now we can understand why.

Gradient descent has a similar principle: if you take steps that are too large, you can overshoot the minimum and oscillate. Smaller steps converge more reliably, though they take longer.

In functional gradient descent, each weak learner $h_m$ approximates the gradient direction but doesn't capture it perfectly. A full step ($\eta = 1$) would add the weak learner at full strength, potentially overcorrecting in some regions.

By using $\eta < 1$, we:

1. Reduce the impact of imperfect gradient approximation
2. Allow subsequent learners to refine the correction
3. Create a smoother path through function space

Friedman's original paper showed empirically that $\eta \in [0.01, 0.1]$ with more iterations typically outperforms $\eta = 1$ with fewer iterations.

> [!note] Shrinkage as Regularization
> Shrinkage is equivalent to early stopping along a regularization path. With smaller $\eta$, the model takes more steps before reaching any given complexity level, allowing early stopping to find a better stopping point.

---

## From First-Order to Second-Order

The algorithm above uses only the **gradient** (first derivative). Modern implementations like XGBoost and LightGBM also use the **Hessian** (second derivative) for better optimization.

The idea is to approximate the loss with a second-order Taylor expansion:

$$
L(y, F + h) \approx L(y, F) + g \cdot h + \frac{1}{2} H \cdot h^2
$$

where:
- $g = \frac{\partial L}{\partial F}$ is the gradient
- $H = \frac{\partial^2 L}{\partial F^2}$ is the Hessian

This gives us more information about the curvature of the loss function. In regions where the Hessian is large (high curvature), we should take smaller steps; where it's small (flat), we can take larger steps.

We'll derive how the Hessian leads to the optimal leaf values and the split gain formula in [[gb-part-3|Part 3]].

For now, here are the Hessians for common losses:

| Loss | Gradient ($g$) | Hessian ($H$) |
|------|----------------|---------------|
| Squared Error | $F - y$ | $1$ |
| Logistic | $p - y$ | $p(1-p)$ |
| Absolute Error | $\text{sign}(F-y)$ | $0$ (see note) |

The Hessian for squared error is constant (1), so first-order and second-order methods are equivalent. For logistic loss, the Hessian $p(1-p)$ is largest when $p \approx 0.5$ (uncertain predictions) and smallest near 0 or 1 (confident predictions).

> [!warning] Absolute Error and Zero Hessian
> The Hessian of absolute error is 0 everywhere (except at $F = y$, where it's undefined). This means second-order methods can't be applied directly. In practice, implementations use a smoothed variant called Huber loss, or they clip the Hessian to a small positive value.

---

## Putting It Into Practice

Let's trace through one boosting round for logistic loss.

> [!example] One Boosting Round (Binary Classification)
> **Setup**: 4 training samples, current predictions $F$, true labels $y \in \{0, 1\}$
>
> | Sample | $y$ | $F$ | $p = \sigma(F)$ | Gradient ($p - y$) | Pseudo-residual ($y - p$) |
> |--------|-----|-----|-----------------|-------------------|---------------------------|
> | 1 | 1 | 0.5 | 0.62 | -0.38 | +0.38 |
> | 2 | 0 | 1.0 | 0.73 | +0.73 | -0.73 |
> | 3 | 1 | -1.0 | 0.27 | -0.73 | +0.73 |
> | 4 | 0 | -0.5 | 0.38 | +0.38 | -0.38 |
>
> **Steps**:
> 1. Compute pseudo-residuals (rightmost column)
> 2. Fit a shallow tree to predict these pseudo-residuals from features
> 3. Add the tree's predictions (scaled by $\eta$) to current $F$
> 4. Repeat

Sample 3 has the largest pseudo-residual (+0.73): a positive example that the model currently thinks is negative ($p = 0.27$). The tree will try to increase predictions for samples that look like sample 3.

---

## What's Next

We've now formalized gradient boosting as functional gradient descent. The pseudo-residual emerges naturally as the negative gradient of the loss, and the algorithm generalizes to any differentiable loss function.

But we've been vague about what "fit a weak learner" actually means. How do we build trees that are good at fitting pseudo-residuals? How do we decide where to split?

The next post, [[gb-part-3|Trees and the Split Gain Formula]], dives into the details:

- Why trees are the standard weak learner
- How the Hessian leads to optimal leaf values
- The split gain formula that powers XGBoost and LightGBM

---

## Summary

**Functional gradient descent** is gradient descent where the "variable" being optimized is a function rather than parameters:

1. Compute the negative gradient of the loss at each training point: the **pseudo-residual**
2. Fit a weak learner to approximate this gradient direction
3. Add the weak learner (scaled by learning rate) to the ensemble
4. Repeat

Key ideas:

- **Pseudo-residual**: $r_i = -\partial L(y_i, F(x_i)) / \partial F(x_i)$
- For squared error, pseudo-residual = residual
- For logistic loss, pseudo-residual = $y - p$
- **Shrinkage** ensures we don't overshoot; smaller $\eta$ = more regularization
- **Second-order methods** use the Hessian for better step sizes

$$
F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)
$$

---

## References

1. Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine". *Annals of Statistics*, 29(5), 1189-1232. [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)

2. Mason, L., Baxter, J., Bartlett, P., & Frean, M. (1999). "Boosting Algorithms as Gradient Descent in Function Space". *NIPS 1999*. [PDF](https://proceedings.neurips.cc/paper/1999/file/96a93ba89a5b5c6c226e49b88973f46e-Paper.pdf)

3. Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System". *KDD 2016*. [arXiv](https://arxiv.org/abs/1603.02754)
