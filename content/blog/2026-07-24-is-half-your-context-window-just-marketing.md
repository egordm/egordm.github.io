---
title: "Is Half Your Context Window Just Marketing?"
date: 2026-07-24
draft: false
tags:
  - llm
  - agents
  - interpretability
  - long-context
description: "A 32,768-token context window is a claim about attention, not memory. We test recall with needles no keyword search or memorized fact can solve, and the model's real operating range comes back at roughly half the number on the box."
aliases:
  - "recall-validity"
  - "effective-context-length"
---

[[blog/2026-07-22-does-a-reasoning-model-actually-read-its-own-thinking-trace|The previous post]] found that a reasoning model can recompute an answer straight from the source text, without needing its own written derivation to survive. The fact doesn't have to live in the trace. It only has to still be reachable in the context. That "still reachable" is the assumption this post tests, because it is not free: a fact sitting far back in a long prompt is not automatically as available as one sitting at the top.

Every model card advertises a context window in tokens. [Qwen3-8B](https://arxiv.org/abs/2505.09388), the model this series keeps coming back to, ships with a native window of 32,768 tokens. That number describes how much text the model's attention mechanism can address in one prompt without truncating. It does not say how many of those tokens a fact planted deep inside remains recoverable from. Those are different claims, and only one of them is on the box.

## Why a naive needle test flatters the model

The standard way to check this is a needle-in-a-haystack test: bury a fact somewhere in a long document, ask a question that needs it, see if the model answers correctly. The trouble is that most of these tests quietly let the model cheat.

If the question and the planted fact share a word, the model doesn't have to reach across the context. It can grep. Ask "what port is the server on" over a haystack that plants "the server's port is 8080," and the word "port" sitting right next to "8080" is a lexical anchor a shallow attention pattern can latch onto at any distance. A test built this way measures whether the model can still find a matching keyword at length, which is a real capability, but it is not recall in the sense a working agent needs: pulling in a fact that was described, not quoted.

The fix, following NoLiMa's construction (Modarressi et al., [arXiv 2502.05167](https://arxiv.org/abs/2502.05167)), is to remove the crutch by design:

- **Zero lexical overlap.** The question's anchor term and the needle's content term share no words and no morphological root; the only link between them is real-world association (Hamlet is Shakespeare's most famous play), never a shared string.
- **An invented entity carries the answer.** The name that actually answers the question is made up, so no amount of memorized world knowledge can supply it. The model has to have genuinely read the sentence.
- **Frozen before any model run.** The 16-item needle set was locked, with a pre-registered selection rule, before a single generation happened. A blind second coder, checking the frozen set against six wellformedness criteria, still caught a leak: one item asked about a "Star Wars fan" and planted a fact about a "fan convention," same word, same sense, exactly the lexical bridge the construction forbids. It was swapped for the next item in a pre-registered replacement queue before either version ever touched the model.
- **Checked against real retrievers, not just eyeballed.** A keyword ranker (BM25) and an embedding-based ranker both ran over every haystack before any model did. Both had to fail to place the needle's sentence in their own top 5 matches; a needle a keyword or embedding search could find on its own would be testing retrieval, not the model's recall.
- **Pass/fail bars fixed before the results existed.** How many items were allowed to fail, and how the curve would have to shift to count as a real length-driven effect, were both written down before the grid ran.

Here is one real item from the frozen set, question, needle, and its deliberately grep-able twin:

```text
QUESTION            "Which character is a Shakespeare devotee?"

NO-OVERLAP NEEDLE    "Greta Simeon can recite Hamlet's soliloquy
(the original)        from memory at the slightest invitation."

LEXICAL-OVERLAP       "Greta Simeon can recite Shakespeare's Hamlet
TWIN                  soliloquy from memory at the slightest
                       invitation."
```

Both sentences encode the same fact. "Greta Simeon" is invented; nothing about Shakespeare tells you who she is, only the sentence does. Only the twin puts the question's own word, "Shakespeare," next to the answer.

## The twin that turns a curve into a measurement

That twin is the control that makes the rest of this post interpretable. Every needle in the set has one: same fact, same entity, but the question's keyword now sits inside the planted sentence. Running both constructions at the same depth and length gives two readings of the same underlying fact:

- The **twin** measures whether the model can still find a fact when it can be grepped, keyword and content in the same sentence.
- The **original** measures whether the model can still make the associative hop when there is no shared word to grep for.

If both curves degrade together with length, the model is losing raw attention capacity across the board. If only the original degrades while the twin holds, something more specific is failing: not the ability to fetch, the ability to connect two things that never shared a surface form.

## Perfect through 8K, 0.757 by 24K

The run: Qwen3-8B, thinking mode on, one RTX 4090, bf16 weights and KV cache, no quantization, stock configuration (no context-extension scaling). 16 needles, each inserted into a natural-document haystack at 5 depths and 5 lengths spanning 1,000 to 24,000 tokens, all inside the native 32,768-token window. 592 generations total. 14 of the 16 needles cleared the usage filter (correct when the fact is easy to reach, wrong when asked cold with no context at all); the two dropped were conservative exclusions, one lost to matcher strictness on a possessive form, one to a borderline base score, neither a recall failure on the item itself.

The depth-aggregated access curve on the 14 used needles:

| context length | access |
|---|---|
| 1,000 tokens | 1.000 |
| 4,000 tokens | 1.000 |
| 8,000 tokens | 1.000 |
| 16,000 tokens | 0.900 |
| 24,000 tokens | 0.757 |

![[blog/assets/recall-access-curve.png|700]]
*The depth-aggregated access curve on the 14 used needles: perfect through 8K, 0.900 at 16K, 0.757 at 24K, crossing the field-standard 85-percent-of-base line between 16K and 24K. The shaded band is the item-bootstrapped 95 percent CI; the green diamonds are the lexical-overlap twins at their two tested endpoint cells, at ceiling (14 of 14, on the same used set as the curve) at both. Qwen3-8B, thinking on, one RTX 4090.*

By the field's own definition (RULER and NoLiMa both use it: the longest tested length still at or above 85 percent of the easy-case baseline), the effective recall length here is 16,000 tokens. The model's advertised window is 32,768. Half the window is marketing.

Before trusting that number, every access-negative cell at 16,000 and 24,000 tokens, 24 of them, was opened and read by hand: is the model actually wrong, or did the matcher just fail to parse a correct answer. All 24 were genuine misses. The model either declined to name a character or named one invented for a different, filler part of the haystack. None were parsing or grading artifacts, and the same transition shows up under the most lenient possible crediting, so the decline isn't a byproduct of how strictly the answer was scored.

## What dies at depth is the hop, not the fetch

The lexical-overlap twin, run at the shallowest/shortest and deepest/longest grid cells only, scores 30 of 32 correct, and both misses trace to the single item already dropped from the main analysis for an unrelated base-score flake. Its longest, deepest cell (24,000 tokens, depth 0.9) sits inside the same length where the no-overlap original's depth-aggregated curve has already fallen to 0.757. Fetching a fact you can point to with a shared word survives the whole tested range. Making the same fetch without a shared word to anchor on is what degrades.

This is the same fingerprint NoLiMa found in the first place, one level down: Llama-3.1-8B, no reasoning mode, falls from a 76.7 percent baseline to 14.2 percent by 32K on this exact no-overlap construction. Different model, different regime, so the two numbers don't average into one line, but the shape matches: a no-overlap needle degrading hard well inside the model's own advertised window, while a lexically-matched version of the identical fact does not. No published run of this construction against a thinking Qwen3 model existed before this one; turning thinking mode on narrows the collapse (0.757, not 0.142, at the longest length tested here) but does not erase it.

One depth pattern is worth flagging because it breaks the naive intuition. At 24,000 tokens, the worst-performing insertion point is not the deepest position in the document (0.86 there); it's the middle (0.43). This matches a known shape change: position bias in long-context models flips from a primacy/recency pattern to a distance-from-the-end pattern once the input crosses roughly half the context window (Veseli et al., [arXiv 2508.07479](https://arxiv.org/abs/2508.07479)), which is exactly the boundary the 16K and 24K cells sit past.

![[blog/assets/veseli-positional-bias-shift.png|460]]
*Accuracy by position of the relevant text as inputs grow toward the context window limit: at short relative lengths the curves dip in the middle (the lost-in-the-middle shape), and as relative length approaches 1.0 the first-position advantage erodes until accuracy simply falls with distance from the end. Figure 1 of Veseli et al., ["Positional Biases Shift as Inputs Approach Context Window Limits"](https://arxiv.org/abs/2508.07479), arXiv 2508.07479 (CC BY 4.0).*

## Where this doesn't reach

- **Static, single-turn only.** Every generation here is one prompt, one answer. A running agent session, tool calls and file reads accumulating turn by turn, is a different construct with its own attention pattern; this probe does not speak to it.
- **One model, one GPU.** Qwen3-8B, bf16, no quantization, a single RTX 4090. The KV cache is also what capped the tested range at 24,000 tokens rather than the full 32,768; that ceiling is a hardware budget, not a claim that recall stops mattering past it.
- **A residual surface-cue confound.** Two independent retrievers, a keyword ranker and an embedding model, confirm the fact isn't findable by lexical or semantic search alone, which rules out the obvious shortcuts. It does not rule out the model keying on some other surface quirk of the invented name rather than genuine reach-back.

## What it means for building agents

The number on a model card describes attention capacity: how large a prompt the model can address without truncating it. It does not describe how reliably a fact planted anywhere in that budget survives to the point where the model needs it. Those are separate properties, and only the first one gets marketed.

If you are filling a long-running agent's context with file reads, tool outputs, and accumulated history, planning around the full advertised window is planning around the wrong number. On this model and this hardware, the reliable range measured at roughly half of it. A few things follow directly from what actually degraded and what didn't:

- **Don't assume a fact read once stays reachable near verbatim.** The decline here is real and measured well inside the advertised window, not a hypothetical edge case at the very end of it.
- **A keyword match recovers what an unaided reach-back loses.** The lexically-matched twin held at every tested length, including the one where the no-overlap original had already dropped to 0.757. Surfacing a fact again near where it's needed, rather than trusting a single early mention to still be reachable, is doing real work, even against a model that reads its entire context.
- **The number is per-model, per-hardware.** Nothing here says another model's curve lands at the same place; it says the number printed on the box is not the number to plan around, and measuring your own is cheaper than assuming.

This is the third of four separate things worth measuring about whether an agent actually uses its context: whether an answer leaned on the right source, whether it read that source correctly, and now, whether it can still reach a fact after enough tokens have piled up since it appeared. A fourth question sits outside what a single static prompt can answer by construction: how that reach holds up not across a longer document, but across the turns of a session that keeps running.

## What the number on the box was measuring all along

A context window's token count is real. It is the largest prompt the model's attention mechanism can address without cutting it off. What it does not promise is that a fact planted anywhere inside that budget is equally recoverable. On this model, on this hardware, with the model's own reasoning doing the retrieving, a needle held perfectly through a quarter of the window and had visibly eroded by three quarters of it, well before the window itself ran out. The version that shared a word with its own question never budged.

If a spec sheet's context window is the only number in your planning, you are planning around capacity. Recall is a separate, smaller number, unadvertised, and on this one model it printed at roughly half.

---

**References.**

- Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze, "NoLiMa: Long-Context Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167).
- Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg, "RULER: What's the Real Context Size of Your Long-Context Language Models?" [arXiv 2404.06654](https://arxiv.org/abs/2404.06654).
- Veseli, Chibane, Toneva, Koller, "Positional Biases Shift as Inputs Approach Context Window Limits," [arXiv 2508.07479](https://arxiv.org/abs/2508.07479).
- Qwen Team, "Qwen3 Technical Report," [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
