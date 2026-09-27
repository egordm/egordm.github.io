---
title: "Is Half Your Context Window Just Marketing?"
date: 2026-07-24
draft: false
series: "LLM Comprehension"
series_order: 4
tags:
  - llm
  - interpretability
  - long-context
description: "Qwen3-8B advertises a 32K-token context window. On questions that share no words with the fact they need, it finds that fact perfectly up to 8K tokens, and its usable range comes out at about half the window."
aliases:
  - "recall-validity"
  - "effective-context-length"
---

Every model card lists a context window. [Qwen3-8B](https://arxiv.org/abs/2505.09388), a popular
open model, lists 32,768 tokens. That number says how much text fits into one prompt. It does not
say whether the model can still *use* a fact that sits somewhere in all that text.

I tested that directly. The short answer: on questions that need the model to connect ideas rather
than match words, Qwen3-8B is perfect up to 8K tokens, slips after that, and its usable range comes
out at about **16K tokens, half the window**.

## Why the usual test flatters the model

The standard check is a needle-in-a-haystack test: hide a fact in a long document, then ask about
it. The catch is that the question usually shares a word with the fact. Ask "what port is the
server on" and the model only has to spot the word *port* next to *8080*. That is a search for a
matching word, and models pass it at almost any length.

The NoLiMa benchmark (Modarressi et al., [arXiv 2502.05167](https://arxiv.org/abs/2502.05167))
removes that shortcut with two rules. The question and the fact share no words, so the only link
between them is knowledge the model has to apply. And the answer is an invented name, so the model
cannot know it from training; it has to have read the sentence. Here is a real item from my test
set:

<img class="theme-dark-only figure-narrow" src="/blog/assets/recall-v-pair-dark.svg" alt="The question 'Which character is a Shakespeare devotee?'. The original needle, 'Greta Simeon can recite Hamlet's soliloquy from memory', needs the hop Shakespeare to Hamlet to Greta. The keyword twin, 'Greta Simeon can recite Shakespeare's Hamlet soliloquy', puts the question's own word next to the answer." />
<img class="theme-light-only figure-narrow" src="/blog/assets/recall-v-pair-light.svg" alt="The same pair of needles in the light theme." />

To answer from the original, the model has to know that Hamlet is Shakespeare's play and link that
to a sentence it read thousands of tokens earlier. The keyword twin carries the same fact, but a
simple word match finds it. Every item in the set has such a twin, and it is the control that makes
the result readable: if the model loses both, it is losing the text; if it loses only the original,
it is losing the connection.

## The experiment

I built 16 such items and hid each one in a long, natural document at five lengths (1K to 24K
tokens) and five positions (from near the start to near the end). Qwen3-8B answered with its
reasoning ("thinking") mode on, on one RTX 4090, with no quantization. Two items failed a check on
short documents that was fixed in advance, for reasons unrelated to recall, and were dropped. That
leaves 14 items and 70 answers per length.

> [!note]- How the test set was protected against shortcuts
> - **Frozen before the first run**, with the rules for dropping and replacing items written down
>   in advance.
> - **A blind second check caught one leak.** An item asked about a "Star Wars fan" and planted a
>   fact about a "fan convention": the same word, the very shortcut the design forbids. It was
>   swapped for the next item in the queue before either version reached the model.
> - **A shortcut check with a real search tool.** A keyword ranker (BM25) searched the 24K
>   documents for each fact's sentence and never placed it in its top 5. A simple word search does
>   not find these facts.
> - **Pass and fail bars set before the results existed.**

## What happened

<img class="theme-dark-only figure-narrow" src="/blog/assets/recall-v-curve-dark.svg" alt="Share of needles found by document length: 100% at 1K, 4K and 8K, 90% at 16K, 76% at 24K, crossing the 85% bar between 16K and 24K, inside the 32K window. The keyword twin was found 14 of 14 times at both cells tested, 1K and 24K." />
<img class="theme-light-only figure-narrow" src="/blog/assets/recall-v-curve-light.svg" alt="The result chart in the light theme." />

**Perfect to 8K, then a slide.** Qwen3-8B found every fact up to 8K tokens, 90% at 16K and 76% at
24K. The field's usual yardstick, used by both NoLiMa and RULER, calls a length usable while the
model keeps 85% of its short-context score. Here that score is 100%, so the bar is 85%: 16K clears
it, 24K does not. The usable range is **16K tokens**, half of the advertised 32K. (The true crossing
lies somewhere between 16K and 24K; those are the lengths I tested.)

**The misses are real.** I read all 24 wrong answers at 16K and 24K by hand. Every one was a genuine
miss: the model either said no character fit, or named a character from the filler text. None was a
grading error.

**The keyword twin held where the original slipped.** At 24K tokens, with the fact near the end of
the document, the twin was still found 14 times out of 14. The twin was only tested at two cells, so
this is a strong hint rather than a full curve, but it points one way: the model can still find the
text; what gets harder is making the connection.

**The middle is the worst place.** At 24K, a fact in the middle of the document was found 6 times
out of 14, against 12 out of 14 near the end: the "lost in the middle" pattern (Liu et al.,
[arXiv 2307.03172](https://arxiv.org/abs/2307.03172)).

The shape matches what NoLiMa itself found on other models. Llama-3.1-8B, without reasoning, fell
from 76.7% on short documents to 14.2% at 32K on the same kind of question.

## What this means when you use an LLM

The number on the model card is capacity: how much text fits. How much of that text the model can
still connect to your question is a separate, smaller number, and nobody prints it.

- **Plan around half the window, not all of it.** On this model the decline starts well inside the
  advertised range, not at the edge.
- **Restate what matters, in the same words.** The keyword twin held where the original slipped.
  Repeating a key fact near where it is needed turns a hard connection into an easy word match.
- **Measure your own model.** These numbers are for one model at one setting. The test is cheap to
  run, and the yardstick above tells you what to read off it.

This covers one question and one answer. A long, multi-turn session adds complications of its own
that this test does not measure.

Does the same hold for a frontier model with a million-token window? I ran the same kind of test on
GPT-6 Luna in
[[blog/2026-09-27-is-half-your-context-window-still-marketing|Is Half Your Context Window Still Marketing in 2026?]]
The window is 30 times larger, the usable part is larger too, and it is still a fraction of what
the model card says.

---

*Revised on 2026-09-27: rewritten for readability, with new figures and a few narrower claims.*

**References.**

- Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze, "NoLiMa: Long-Context Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167).
- Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg, "RULER: What's the Real Context Size of Your Long-Context Language Models?" [arXiv 2404.06654](https://arxiv.org/abs/2404.06654).
- Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang, "Lost in the Middle: How Language Models Use Long Contexts," [arXiv 2307.03172](https://arxiv.org/abs/2307.03172).
- Qwen Team, "Qwen3 Technical Report," [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
