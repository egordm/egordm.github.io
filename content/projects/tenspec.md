---
title: "Tenspec"
description: "Declare what an array must be, beside the value it describes, and check it at one boundary."
tags:
  - "python"
  - "open-source"
  - "typing"
---

**Tenspec** lets you declare what an array must be, beside the value it describes, and check it at one boundary.

The shape and dtype checks at the top of a function move into its signature, where a reader and a type checker both see them. Tenspec accepts or refuses the array the caller passed: it converts nothing, moves nothing, and copies nothing unless the declaration names a transform that copies.

## What it gives you

- **Ordinary types.** A declaration is an ordinary annotation carrying the array type of your backend (NumPy or PyTorch). Code that never calls Tenspec still imports and runs.
- **Pydantic boundaries.** `TensorContracts` relates the fields of your own model inside one model validation.
- **Runtime checks are opt in.** An annotation alone does no runtime work; `@checked`, `TensorContracts` or `validate` decide where a value is checked.

## Quick start

```bash
pip install "tenspec[numpy]"
```

```python
from typing import Literal as Shape

import numpy as np

from tenspec import checked
from tenspec.numpy import Float


@checked
def weigh_columns(
    values: Float[Shape["rows features"]], weights: Float[Shape["features"]]
) -> Float[Shape["rows features"]]:
    return values * weights


print(weigh_columns(np.ones((3, 4)), np.array([1.0, 2.0, 3.0, 4.0])).shape)
```

`weights` must cover as many features as `values` has columns, and the result must keep its shape. Tenspec is an early release (0.1.0) and needs Python 3.13 or newer.

## Links

- **[Documentation](https://egordmitriev.dev/tenspec/)**: the guide, the API reference and two executed tutorials
- **[GitHub repository](https://github.com/egordm/tenspec)**: source code and issue tracker

## Related posts

The [[series/programming-with-invariants|Programming with Invariants]] series explains the ideas behind Tenspec, and its second part uses it:

1. [[blog/let-your-types-carry-the-rules|Let Your Types Carry the Rules]]
2. [[blog/when-your-types-are-arrays|When Your Types Are Arrays]]
