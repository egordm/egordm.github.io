---
title: "Programming with Invariants"
description: "Model meaningful inputs, check them once at the boundary, and let the rest of the code do its job. From plain values to arrays with named axes."
---

Functions that should implement an algorithm often fill up with defensive checks: is this count positive, do these timestamps have timezones, does the end come after the start. This series moves those rules into the values themselves, checked once where data enters, so the code behind the boundary can rely on them. Each post works through one example.

## Posts in this series

1. **[[blog/let-your-types-carry-the-rules|Let Your Types Carry the Rules]]** - Move repeated input checks into meaningful values, then let the rest of the program do its job. A worked example with counts, timezones, and a report window.
2. **[[blog/when-your-types-are-arrays|When Your Types Are Arrays]]** - Named axes help explain tensor code; checked boundaries make those descriptions useful guarantees. A worked example from an attention-head calculation.
