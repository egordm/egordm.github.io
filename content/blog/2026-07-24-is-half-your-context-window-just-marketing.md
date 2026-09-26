---
title: "Is Half Your Context Window Just Marketing?"
date: 2026-07-24
draft: false
series: "LLM Comprehension"
series_order: 6
tags:
  - llm
  - interpretability
  - long-context
description: "How much context can an LLM use when matching words will not find the answer? Qwen3-8B clears the recall bar at 16K on my 14-item set, about half its native window. Other sets put it lower."
aliases:
  - "recall-validity"
  - "effective-context-length"
---

The context window tells us how much text fits into a prompt. It leaves open the question we care about: **can the model find the fact our question needs?** That becomes harder to test when the question and the fact use different words.

I tested Qwen3-8B, which has a native window of **32,768 tokens (131,072 with YaRN, which I did not use)**. On my set, recall stayed usable to about **16K**, roughly half the native window. That is a result from one small experiment, not a rule for how much context to trust. Let us build the test before reading the curve.

## Take away the matching word

Suppose we hide a fact inside a long document and ask about it at the end. If the question repeats a distinctive word from the fact, the model can use that word to locate the answer. To test whether it can find the fact by meaning, we remove that shared clue.

The [NoLiMa benchmark](https://arxiv.org/abs/2502.05167) uses questions that share no words with the facts they need. The answer is an invented name, so knowing the subject is not enough: the model must connect that knowledge to a name in the document.

Here is an item I authored using that construction:

> **Fact:** Greta Simeon can recite Hamlet's soliloquy from memory at the slightest invitation.
>
> **Question:** Which character is a Shakespeare devotee?

We can connect Shakespeare to Hamlet from knowledge of the play. But only the document connects Hamlet to Greta Simeon. For the **keyword version**, I add Shakespeare's name to the fact. Now the question supplies a word the model can match directly. The diagram shows the relation the question requires, not a measured path inside the model.

<img class="theme-dark-only figure-center" src="/blog/assets/recall-v-pair-dark.svg" alt="Construction of the authored Greta Simeon example. By meaning: Shakespeare connects to Hamlet, which connects to Greta Simeon. With a shared word: Shakespeare connects directly to Greta Simeon. This item was later dropped from scoring." />
<img class="theme-light-only figure-center" src="/blog/assets/recall-v-pair-light.svg" alt="Construction of the authored Greta Simeon example. By meaning: Shakespeare connects to Hamlet, which connects to Greta Simeon. With a shared word: Shakespeare connects directly to Greta Simeon. This item was later dropped from scoring." />

This example belongs to the authored set, but **was later dropped from scoring**: the model recovered the fact, and the strict answer matcher missed the possessive form “Hamlet's”. It illustrates the construction; it contributes nothing to the results below.

Every item had a keyword version. This is the **control**: it asks whether the model can recover the answer when a matching word helps it find the fact. If that version holds while the version without shared words declines, the difficulty depends on how we ask the model to retrieve the fact. That comparison alone cannot tell us where inside the model the loss occurs.

## Build the document, then move the fact

I authored 16 items and placed each fact in a long, natural document. I varied the document length across 1K, 4K, 8K, 16K and 24K tokens, and the fact's position from near the start to near the end. Each item appeared at five positions: 10%, 30%, 50%, 70% and 90% of the way through the document.

I ran Qwen3-8B with thinking on, using one RTX 4090 and no quantization. Alongside the possessive-matcher item, I dropped an item whose short-document answers were inconsistent across repeated draws. The rules for dropping and replacing items were fixed before the run. That left **14 scored items, with one draw per item at each length and position: 70 answers per length**.

Here, **found** means the model's reasoning restates the fact with its key word in the same sentence. That is a stricter measure than checking only the final answer, which scored higher. The curve below measures this fact-recovery score.

> [!note]- Checks against shortcuts
> I froze the items and the pass and fail bars before the run. A blind second check caught a repeated word in an item, which I replaced before either version reached the model.
>
> At 24K, with the fact in the middle, the keyword search tool BM25 put none of the 16 authored facts in its top 5 results. That check covered this position only. A meaning-based search tool, Contriever, ranked every fact first at every tested length. The facts were hard to find by matching words, but remained retrievable by meaning.

## How much recall counts as usable?

We need a bar before we can turn a recall curve into a length. **Effective length** is the longest tested length that keeps at least **85% of the short-context score**, following the yardstick used by NoLiMa and RULER. It is a tolerance for a drop in performance, not a guarantee that every question will work.

My short-context score was 100%, so the bar here is 85%. Recall stayed at 100% through 8K, then fell to 90% at 16K and about 76% at 24K. The dashed line lets us read off which tested lengths clear the bar. The shaded region beyond 24K contains no measurements.

<img class="theme-dark-only figure-center" src="/blog/assets/recall-v-curve-dark.svg" alt="Fact recovery averaged over five positions: 100% at 1K, 4K and 8K, 90% at 16K, and about 76% at 24K. The 85% bar lies between the 16K and 24K results. The region from 24K to the native limit of 32,768 tokens is untested." />
<img class="theme-light-only figure-center" src="/blog/assets/recall-v-curve-light.svg" alt="Fact recovery averaged over five positions: 100% at 1K, 4K and 8K, 90% at 16K, and about 76% at 24K. The 85% bar lies between the 16K and 24K results. The region from 24K to the native limit of 32,768 tokens is untested." />

**16K clears the bar; 24K does not.** The tested points put the crossing between them. The effective length is therefore 16K on this set, about half the native window. I checked all 24 wrong answers at 16K and 24K: the model either declined to identify a character or named one from the surrounding text. None was a matcher or parsing error.

The scope matters. This is **one model, 14 items, one draw per cell**. On an extended 34-item set with thinking on, 16K already fell below the bar. On NoLiMa's own items with greedy thinking, recall fell from 98% at 1K to 69% at 16K. Other item sets put the usable length lower.

## What the control tells us

The keyword version recovered **14 of 14 facts at both tested cells**: 1K near the start and 24K near the end. These are two checks, not a full curve.

**Matching by shared words still works at 24K; finding the fact by meaning is what gets harder.** We cannot conclude that the model first finds the text and then fails only to make the connection. Later probes disagree: some suggest a weakened connection that loses to competing answers, others suggest the connection was never made. Further attempts could not reliably separate finding the fact from using it. The control establishes a difference in behaviour, not a single mechanism.

## Position changes the result too

The curve averages over positions. At 24K, that hides a large difference: the model found **6 of 14 facts in the middle**, versus **12 of 14 near the end**. Below, the narrow marker places the fact within the same document length; each dot represents a scored item, filled when its fact was found.

<img class="theme-dark-only figure-center" src="/blog/assets/recall-v-position-dark.svg" alt="At 24K tokens, a fact placed halfway through the document was found in 6 of 14 items. At 90% of the way through, near the end, it was found in 12 of 14. Equal-length document bars mark the two positions; filled dots show recovered facts." />
<img class="theme-light-only figure-center" src="/blog/assets/recall-v-position-light.svg" alt="At 24K tokens, a fact placed halfway through the document was found in 6 of 14 items. At 90% of the way through, near the end, it was found in 12 of 14. Equal-length document bars mark the two positions; filled dots show recovered facts." />

This resembles the [Lost in the Middle](https://arxiv.org/abs/2307.03172) result, but the middle is not always the weakest position. At 16K, the weakest position was 30% of the way through. Moving a fact nearer the question can help; no position guarantees recovery.

## What to do with your own context

Capacity and usable recall answer different questions. NoLiMa and RULER publish effective lengths, but **no model card prints the usable number for your task**. We need to measure it on the questions and documents we actually use.

I would start with facts the model can recover in short documents, then add surrounding text and move those facts around. Include questions that use different words from the source, alongside versions that repeat the key terms. Choose an acceptable drop before reading the results. That gives us a working limit grounded in the task.

For important facts, make the connection explicit and keep the relevant text close to the question. The controls give a reason to try this, not a promise that every long prompt will work. This experiment covers one question and one answer; a long conversation needs its own checks.

**Do not budget a fixed half of the window.** In [[blog/2026-09-27-is-half-your-context-window-still-marketing|Is Half Your Context Window Still Marketing in 2026?]], I test GPT-6 Luna with a window about 30 times larger. Its usable range is about 64K with reasoning off and 128K with reasoning high, roughly 6% to 13% of its advertised 1M. A larger window can give us more usable context while leaving a smaller usable share. Measure that share for your workload, then keep the facts your answer depends on within it.

---

*Revised on 2026-09-27: new figures, the scope of the 16K result stated, and a softer reading of why recall drops.*

**References.**

- Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze, "NoLiMa: Long-Context Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167).
- Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg, "RULER: What's the Real Context Size of Your Long-Context Language Models?" [arXiv 2404.06654](https://arxiv.org/abs/2404.06654).
- Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang, "Lost in the Middle: How Language Models Use Long Contexts," [arXiv 2307.03172](https://arxiv.org/abs/2307.03172).
- Qwen Team, "Qwen3 Technical Report," [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
