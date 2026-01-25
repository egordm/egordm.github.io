---
title: "Inside Gradient Boosting"
description: "A deep dive into how XGBoost and LightGBM work — from first principles to implementation details."
---

A comprehensive series explaining gradient boosting from first principles to advanced implementation details. Learn how modern gradient boosting libraries like XGBoost and LightGBM actually work under the hood.

## Posts in this series

1. **[[blog/2026-01-06-what-is-gradient-boosting|What is Gradient Boosting?]]** — Intuition for gradient boosting without heavy math
2. **[[blog/2026-01-08-functional-gradient-descent|Functional Gradient Descent]]** — The mathematical foundation: gradient descent in function space
3. **[[blog/2026-01-10-trees-and-split-gain|Trees and the Split Gain Formula]]** — Why trees are the standard weak learner and how we derive optimal splits
4. **[[blog/2026-01-12-histogram-based-split-finding|Histogram-Based Split Finding]]** — How XGBoost and LightGBM achieve O(bins) complexity
5. **[[blog/2026-01-14-depth-wise-vs-leaf-wise|Depth-Wise vs Leaf-Wise Growth]]** — Two strategies for tree construction
6. **[[blog/2026-01-16-gradient-based-sampling|Gradient-Based Sampling (GOSS)]]** — LightGBM's sampling optimization
7. **[[blog/2026-01-18-efb-and-categorical-features|EFB and Categorical Features]]** — Exclusive Feature Bundling and categorical handling
8. **[[blog/2026-01-20-regularization|Regularization in Practice]]** — Hyperparameter tuning for gradient boosting
9. **[[blog/2026-01-22-xgboost-vs-lightgbm|XGBoost vs LightGBM]]** — A practical comparison
