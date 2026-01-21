---
title: "Boosters"
description: "A high-performance gradient boosting library for Python and Rust"
tags:
  - "machine learning"
  - "rust"
  - "python"
  - "open-source"
---

**Boosters** is a high-performance gradient boosting library for Python and Rust.

## The Story Behind Boosters

What started as a learning experiment turned into a deep dive down the rabbit hole of gradient boosting internals.

I initially set out to **dissect XGBoost models** - understanding what's really inside them and trying to reproduce their results from scratch. This curiosity led me through extensive research into how various optimizations and algorithms actually work under the hood.

Along the way, I documented my entire research and design process on the library's website, complete with proper LaTeX formulas and detailed explanations:

- 📚 **[Research Documentation](https://egordmitriev.dev/boosters/research/index.html)** - Deep dives into gradient boosting theory and algorithms
- 🏗️ **[Design RFCs](https://egordmitriev.dev/boosters/design/rfcs.html)** - Architecture decisions and implementation rationale

I implemented everything in **Rust**, which naturally led me further down the optimization path. After months of work, I ended up with a library that not only has **feature and algorithm parity with XGBoost and LightGBM**, but is also **just as fast - or faster**.

## Features

- ⚡ **High Performance** - Optimized Rust implementation with zero-cost abstractions
- 🐍 **Python Integration** - Seamless Python bindings via PyO3
- 🔧 **Full Feature Parity** - Supports all major XGBoost and LightGBM algorithms
- 📊 **Well Documented** - Comprehensive research docs and design rationale

## Quick Start

```bash
pip install boosters
```

```python
from boosters import GradientBoosting

model = GradientBoosting()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Links

- **📚 [Documentation](https://egordmitriev.dev/boosters/)** - Full docs, tutorials, and API reference
- **🐙 [GitHub Repository](https://github.com/egordm/boosters/)** - Source code and issue tracker
- **🔬 [Research](https://egordmitriev.dev/boosters/research/index.html)** - Theory and algorithm deep-dives
- **🏗️ [Design RFCs](https://egordmitriev.dev/boosters/design/rfcs.html)** - Implementation decisions

## Related Posts

Check out related articles about machine learning:

- [[blog/comprehensive-introduction-to-large-language-models/|Introduction to Large Language Models]]
