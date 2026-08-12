---
title: "Does Precision Loss Break Retrieval, or Does Distance?"
date: 2026-08-12
tags:
  - llm
  - agents
  - interpretability
  - quantization
  - long-context
description: "A quantization ladder and a length ladder, run on the same model with the same task (no shared words between question and fact): precision survives down to roughly 3.5 bits per weight, distance does not survive the model's own native window."
aliases:
  - "dose-vs-distance"
  - "precision-vs-distance"
---

[[blog/2026-07-24-is-half-your-context-window-just-marketing|The previous post]] measured one way a model loses its grip on a fact already sitting in its context: distance. Plant a fact early in a long prompt, ask a question that shares no words with it, and the model's chance of making that connection drops well before the prompt runs out. There is a second, entirely separate way the same fact could be lost: precision. Shrink every weight from bf16 down toward 2 bits and, somewhere on that ladder, the network should stop being able to make the same connection.

Which one actually breaks it first? We ran both axes on the same model, Qwen3-8B with thinking mode on, and this post is the answer.

## The task: two facts, no words to grep

Part 4 planted one fact per question. This post's task is one step harder: each question needs **two** planted facts, combined. Here is a real item from the frozen set:

```text
FACT ONE       "Emrik Vasland brings up the hammer Mjolnir in
(planted)       nearly every conversation."

FACT TWO       "Emrik Vasland signs letters with a made-up name
(planted)       instead of the real one, Cass Orbell."

QUESTION       "Someone here is forever discussing Norse mythology.
                What pseudonym belongs to that individual?"
```

The right answer is "Cass Orbell", and nothing on the way there is a word match. The question says *Norse mythology*; the fact says *Mjolnir*. You only connect them if you know the hammer belongs to that mythology. The question says *pseudonym*; the fact spells out "a made-up name instead of the real one". You only connect those if you know what a pseudonym is. Both names are invented, so no memorized knowledge can supply the answer, and both facts sit planted in a long haystack of unrelated text alongside 23 decoy characters doing similar but wrong things. As in part 4, the model cannot grep its way to the answer. It has to find both facts through meaning and put them together.

Every item also carries a simpler control question over the same haystack: *"Someone here is forever discussing Norse mythology. Identify that individual."* One connection instead of two; the answer is "Emrik Vasland". This control is what makes failures readable. If the control holds while the two-fact score falls, combining is what broke. If both fall, the model is no longer finding facts through meaning at all.

The bank is 38 such items, frozen before any run. The strict score counts a run correct only when the model names exactly the right invented name; a looser check (the right name appearing anywhere in the answer) is reported beside it.

## Eight precision rungs, one real drop

We served the same 38 items at a fixed 4,000-token context on every rung of a GGUF quantization ladder, bf16 down through Q2_K, each rung compared to bf16 item by item. At full precision the model solves about a third of the runs (0.322): the join is hard even with everything close by.

![[blog/assets/latent-dose-curve.png|700]]
*The two-fact join across the quantization ladder at 4K context: flat from BF16 through Q4_K_M, one real drop at Q3_K_S, and a Q2_K score that mostly reflects the model refusing to answer. Qwen3-8B, thinking on, one RTX 4090.*

From Q8_0 down through Q4_K_M, nothing moves outside noise. Q3_K_M, at about 3.4 bits per weight, also stays within noise, just barely. The first real damage is Q3_K_S: the strict score drops by 0.213, while the looser check *rises* by 0.322 at the same rung. The model names more people, with more confidence, and is less often right about which one. At the bottom, Q2_K's score of 0.009 does not measure ability at all: the model refused to answer 76.6 percent of the time, past the run's own 50 percent validity ceiling, so that rung reads "stopped answering", not "answered wrong".

A small pilot ahead of the ladder predicted three failure shapes: at Q3_K_M the model finds the right person but hesitates to name it, at Q3_K_S it names the wrong one confidently, at Q2_K it stops answering. The main run confirmed the last two; the first stayed within noise.

So on this task, quantization is innocent until roughly 3.5 bits per weight, which is well below anything people actually deploy. And when it finally breaks, it breaks confident and wrong, not uncertain.

## Distance breaks it at full precision

The second axis needs no new experiment: [[blog/2026-07-24-is-half-your-context-window-just-marketing|part 4]] already measured it on the same model, at full precision, with the one-fact version of the same task. Perfect recall through 8,000 tokens, 0.900 at 16,000, 0.757 by 24,000, all comfortably inside the native 32,768-token window. No quantization anywhere in the pipeline; ordinary distance does the damage on its own.

![[blog/assets/latent-length-ladder.png|700]]
*Retrieval against context length, at full precision, inside the native 32,768-token window. The upper curve is the 1-fact task: perfect through 8K, 0.900 at 16K, 0.757 at 24K. The lower two series are the harder 2-fact join and its 1-hop control; they sit lower and stop near 16K. The shaded band marks where those harder tasks collapse to zero (an estimate from single runs, not a measured rate) while the easier task is still declining through it. Qwen3-8B, thinking on, one RTX 4090.*

The two-fact join tells the same story on its own length axis. Only its shortest window, around 4,000 tokens, clearly passes the bar set before the run; at 8,000 the average is still well above the bar, but the noise band dips below it. And in single spot-checks, even the one-connection control collapses to zero somewhere between 16,000 and 24,000 tokens, inside the model's own window. A harder task fails earlier, but the shape is the same: gradual decline, then collapse, well before the advertised window runs out.

## Two axes, opposite verdicts

Same model. Same task family. Opposite verdicts.

Push precision down and the join barely moves until Q3_K_S, about 3.2 bits per weight, a level nobody runs in practice; Q4_K_M and Q5_K_M, the settings people actually deploy, sit flat with the full-precision reference. Push length out instead and the same kind of task is already degrading by 16,000 tokens, less than half of this model's native window, and collapses entirely before the window's edge. One axis costs you almost nothing until far past where anyone actually quantizes. The other starts costing you well inside the box printed on the model card.

If you had to bet on which one silently corrupts a long-running agent's grip on its own context, precision loss is not the axis to worry about. Distance is.

## Someone measured this before us

We did not discover that meaning-based recall decays with length. [NoLiMa](https://arxiv.org/abs/2502.05167) (Modarressi et al., ICML 2025) published exactly this shape first, across a range of production models, using the same no-shared-words task design this whole series builds on. Our numbers land beside theirs as an independent check: a different model, our own harness, the same collapse.

The paper's own published table, plus the newest updates its maintainers have posted to the project's repository:

| model | claimed window | effective length | ratio |
|---|---|---|---|
| Gemini 1.5 Pro | 2,000,000 | 2,000 | 0.1% |
| Claude 3.5 Sonnet | 200,000 | 4,000 | 2% |
| GPT-4o | 128,000 | 8,000 | 6% |
| GPT-4.1 | 1,000,000 | 16,000 | 1.6% |
| Llama 4 Scout | 10,000,000 claimed (trained to 256,000, per Meta's own launch blog) | 1,000 | 0.01% |
| Qwen3-8B, thinking on (this series) | 32,768 (native) | 16,000 | 49% |

"Effective length" is each source's own definition: the longest tested length still at or above 85 percent of the easy-case baseline, the same rule part 4 used for this model's own number.

Two things stand out. First, the effective lengths cluster in the same low band, a few thousand to 16K tokens, whether the claimed window is 128,000 or 10,000,000. Llama 4 Scout claims a window 50 times Claude 3.5 Sonnet's, and its effective length is smaller, not larger. How far a model can reach a fact through meaning looks like a property of its training and its internals, not of how far its window has been stretched. Second, our own ratio is the outlier in the table, and not because this model holds facts at distance better than GPT-4.1. It never claimed a window bigger than 32K to begin with, and both models land at the same 16,000-token effective length. A smaller claimed window just makes the same gap look better as a percentage.

One caveat, stated plainly: no 2026-generation frontier model, no GPT-5, no Claude 4.x, no Gemini 3, has a published number on this kind of task that we could find. The NoLiMa project's own repository has not added a model since mid-2025. The pattern holds for every 2025-generation model it has been run on; whether it still holds at the frontier is, right now, unmeasured.

## What it means for agents at long context

A coding agent working against a 200,000 or 1,000,000-token window is not protected by that number the way the number implies. This series has now reached the same conclusion from two directions: an advertised context window says how much text the model can attend over, not how far away a fact can sit and still be reachable by anything other than a keyword match. A tool that re-reads a file it already opened three turns ago, instead of trusting its own earlier summary, is not just being cautious. On this evidence it is closer to correct: past some point well inside the window, a fact that has not been brought back near where it is needed is not reliably there to be recalled at all.

Precision, next to that, is close to free. Quantize hard for cost and the model's grip on a fact it has not quoted word for word stays close to where it started, until deep into settings nobody actually uses. The under-measured axis this series keeps finding is distance, not precision: not whether the weights are sharp enough to hold an idea, but whether the idea is still close enough, in tokens, to be found.

---

**References.**

- Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze, "NoLiMa: Long-Context Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167).
- Adobe Research, [NoLiMa GitHub repository](https://github.com/adobe-research/NoLiMa), 2025 model-generation updates.
- Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg, "RULER: What's the Real Context Size of Your Long-Context Language Models?" [arXiv 2404.06654](https://arxiv.org/abs/2404.06654).
- Meta AI, ["Llama 4: Leading Intelligence"](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) launch blog (pretraining sequence length up to 256K, against the 10M claimed window).
- Qwen Team, "Qwen3 Technical Report," [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
