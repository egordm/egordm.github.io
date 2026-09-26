---
title: "Does Precision Loss Break Retrieval, or Does Distance?"
date: 2026-08-12
series: "LLM Comprehension"
series_order: 7
tags:
  - llm
  - agents
  - interpretability
  - quantization
  - long-context
description: "On Qwen3-8B, the thinking-off join survives quantization to about 3.4 bits per weight, with first damage at about 3.2. With thinking on, retrieval falls as the context grows."
aliases:
  - "dose-vs-distance"
  - "precision-vs-distance"
---

[[blog/2026-07-24-is-half-your-context-window-just-marketing|Part 6]] asked whether an LLM can find a fact through meaning when the question shares no words with it. As the context grew, recall fell. But length is not the only thing that might weaken a connection. **Quantization** stores the model's weights with less precision. Could rounding those learned numbers break retrieval before distance does?

I tested both axes on Qwen3-8B. **The quantization ladder ran with thinking off; the join length curve ran with thinking on.** They are separate experiments in different regimes, not a controlled contest between precision and distance. Let us first see what the model had to connect, then what each experiment establishes.

## Two facts, two connections

A question can point to a fact without repeating its words. Here is an **invented example**, separate from the measured items:

> Lira Venn keeps sketching Saturn.
>
> Lira Venn signs letters as Sorel Ash.
>
> **Question:** Which alias belongs to an astronomy enthusiast?

We first connect *astronomy enthusiast* to someone sketching Saturn. That identifies Lira Venn. We then connect *alias* to the name used for signing letters, Sorel Ash. The question shares no words with either fact. The facts share the person's name, which lets us join them. Both names are invented: general knowledge can connect the meanings, but cannot supply the answer.

The diagram separates those steps. A **one-connection control** asks only for the person associated with astronomy. It stops at Lira Venn; the full question needs the second fact too.

<img class="theme-dark-only figure-center" src="/blog/assets/precision-v-task-dark.svg" alt="Invented example: astronomy enthusiast connects through Saturn to Lira Venn, then through the signing name to Sorel Ash. The control stops at Lira Venn; the join uses both facts." />
<img class="theme-light-only figure-center" src="/blog/assets/precision-v-task-light.svg" alt="Invented example: astronomy enthusiast connects through Saturn to Lira Venn, then through the signing name to Sorel Ash. The control stops at Lira Venn; the join uses both facts." />

I froze a bank of 38 items before running the experiment. Each places its facts in a **haystack**, the surrounding material, alongside unrelated text and 23 decoy characters. Each also has a control question over the same material. If the control holds while the join falls, combining the facts becomes a suspect. If both fall, the failure reaches beyond combining them.

Before reading the scores, we need to distinguish two ways an answer can count. The **strict score** requires a match to the item's accepted answers and allows at most two distinct names from its roster. The **loose score** only checks whether an accepted name appears anywhere in the generated response. A response can therefore pass the loose check while naming too many candidates to pass the strict one.

## How far can precision fall?

A model's weights are learned numbers. Quantization represents them using fewer bits, so nearby values may have to share a stored value. **Bits per weight** describes the average storage used for each weight. Fewer bits mean a smaller representation with coarser precision.

A **rung** is one storage format in this progression. BF16 is the reference; names such as Q4_K_M identify quantization formats, not exact average bit counts. I served the same items at a fixed 4,000-token context on eight rungs, comparing each with BF16 item by item. **Thinking was off throughout this ladder.** At BF16 the strict score is 0.322, about a third: the join is already difficult with the facts nearby.

Read the next figure from left to right as precision decreases. The points show strict scores. The cross marks a rung that failed the experiment's answer-validity check.

<img class="theme-dark-only figure-center" src="/blog/assets/precision-v-dose-dark.svg" alt="Qwen3-8B with thinking off at 4,000 tokens. Strict scores across BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q3_K_S and Q2_K are 0.322, 0.336, 0.322, 0.327, 0.292, 0.246, 0.108 and 0.009. Q2_K is invalid because abstention is 0.766." />
<img class="theme-light-only figure-center" src="/blog/assets/precision-v-dose-light.svg" alt="Qwen3-8B with thinking off at 4,000 tokens. Strict scores across BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q3_K_S and Q2_K are 0.322, 0.336, 0.322, 0.327, 0.292, 0.246, 0.108 and 0.009. Q2_K is invalid because abstention is 0.766." />

From Q8_0 through Q4_K_M, the paired score drops are within noise: their uncertainty intervals include zero. Q3_K_M, at about **3.4 bits per weight**, still has a drop consistent with zero, though only narrowly. This does not establish that its score is identical to BF16.

The first clear damage is **Q3_K_S, about 3.2 bits per weight**. Its strict score falls to 0.108, a paired drop of 0.213. The loose score also **falls**, by 0.322, more than the strict score does. The right name appears less often even under the permissive check. The failure is a commitment to a wrong name, not merely reluctance to choose the correct one.

Q2_K needs a different reading. Its score is 0.009, but its abstention rate is 0.766, beyond the experiment's 0.5 validity ceiling. I cannot interpret that point as a clean retrieval measurement: the model mostly stopped answering.

On this thinking-off task, retrieval therefore **survives down to about 3.4 bits per weight; the first detected damage is at about 3.2**. The result supports those tested settings, not a claim that quantization is harmless for every task.

## What changes when the context grows?

Now keep the weights at BF16 and add material around the facts. I ran the join and its one-connection control **with thinking on**. The control matters here: it lets us ask whether distance hurts just the combination or also the simpler connection.

We also need a reference for guessing. The **chance floor** is the task's registered guessing benchmark, 0.1619. A point below that line does not demonstrate useful retrieval by this criterion. It does not mean the model never answers correctly. I compare point rates with that floor, not uncertainty bounds.

The next figure includes the older one-fact curve from part 6. It comes from a **different authored item set**, not from removing a fact from this join task. Its higher scores help show why the choice of task matters. Whiskers on the join and control show the **strict to permissive range**: count undefined answers as misses or as hits. They are not the spread across repeated runs. The central points score only defined answers.

<img class="theme-dark-only figure-center" src="/blog/assets/precision-v-length-dark.svg" alt="BF16 length curves with thinking on. The join and its one-connection control both cross the 0.1619 chance benchmark between 15,360 and 23,161 haystack tokens. Whiskers show strict to permissive scoring ranges. The older one-fact curve uses different items and one draw per cell; only join and control use nine draws per item." />
<img class="theme-light-only figure-center" src="/blog/assets/precision-v-length-light.svg" alt="BF16 length curves with thinking on. The join and its one-connection control both cross the 0.1619 chance benchmark between 15,360 and 23,161 haystack tokens. Whiskers show strict to permissive scoring ranges. The older one-fact curve uses different items and one draw per cell; only join and control use nine draws per item." />

The join falls from **0.538 at 4,096 haystack tokens** to 0.446 at 7,558, 0.306 at 15,360, and **0.066 at 23,161**. Its strict to permissive ranges are respectively 0.523 to 0.550, 0.436 to 0.459, 0.298 to 0.325, and 0.064 to 0.094.

The control falls too: 0.721, 0.522, 0.269, and 0.092 at those same lengths. Both cross the 0.1619 benchmark between **15,360 and 23,161 haystack tokens**. This weakens the explanation that only joining two facts broke. Even the simpler question over the same material loses its connection.

The older one-fact curve remains perfect through 8,000 tokens, then reaches 0.900 at 16,000 and 0.757 at 24,000. That result uses one draw per cell on a different set: 16 authored items, of which 14 were used. Only the join and its paired control aggregate nine draws per item at each length.

> [!note]- How I held the join length experiment fixed
> Each length cell contains 38 join items and 38 control questions, with nine draws each: 684 generations, or 342 per task. The construction seed holds the material fixed; draw seeds vary the sampled response. I held the following settings throughout **this length experiment**: Qwen3-8B, BF16 GGUF, thinking on, temperature 0.6, top_p 0.95, top_k 20, an output budget of 8,192 tokens, and one backend. I scored accepted answer forms on the answer segment. These settings do not describe the thinking-off quantization ladder or the older one-fact experiment, which used Transformers in BF16.

The horizontal axis counts **haystack material**, not the entire served window. At the last join point, 23,161 material tokens plus the prompt and reasoning budget require a served window of **31,718 tokens**. That is near the native 32,768-token limit. The crossing occurs before that window runs out once the reasoning budget is counted; it is not evidence that the model fails with half its native window still unused.

## What the two experiments let us conclude

The same model gives different outcomes under **two regimes**. With thinking off at a fixed short context, Q4_K_M and Q5_K_M have no detectable loss against BF16 on this join. With thinking on at BF16, the join and its control decline as the haystack grows. These curves do not isolate precision and distance under identical conditions.

There is a useful check with thinking on: a separate **one-fact quantization experiment** of mine, using 34 items at 4K and 8K, found access unchanged down to Q3_K_M, about 3.4 bits per weight. At Q2_K, generation behaviour failed. This supports the narrower conclusion that moderate quantization can preserve access even with reasoning enabled. It does not replace a thinking-on dose experiment on the join itself.

Reasoning also helps with distance. On the join, turning thinking on roughly doubled the haystack the model could still use. Yet the thinking-on length curve still falls. The reasoning setting belongs in the result, alongside the task and the amount of material.

## How this compares with NoLiMa

[NoLiMa](https://arxiv.org/abs/2502.05167), by Modarressi and colleagues at ICML 2025, measured meaning-based recall across models using questions without shared words. My experiments are an independent check with my own harness and items. The table combines its published and repository results with this series' older one-fact measurement.

**Effective length** means the longest tested length still at or above 85 percent of the source's baseline. It is a task-specific measurement, not a universal limit on what a model can read.

| Model | Claimed window | Effective length | Ratio |
|---|---|---|---|
| Gemini 1.5 Pro | 2,000,000 | 2,000 | 0.1% |
| Claude 3.5 Sonnet | 200,000 | 4,000 | 2% |
| GPT-4o | 128,000 | 8,000 | 6% |
| GPT-4.1 | 1,000,000 | 16,000 | 1.6% |
| Llama 4 Scout | 10,000,000 (trained to 256,000, per Meta) | 1,000 | 0.01% |
| Qwen3-8B, thinking on (my part 6 item set) | 32,768 native (131,072 with YaRN) | 16,000 | 49% of native; 12% with YaRN |

For the **2025-generation models tested**, effective lengths stayed in the 1K to 16K band despite much larger differences in claimed windows. Llama 4 Scout claimed 50 times Claude 3.5 Sonnet's window but had a smaller effective length. A larger window did not reliably imply better meaning-based recall in this set; the comparison does not establish training as the cause.

The Qwen3-8B row needs particular care. Its 16,000-token result belongs to **the one-fact item set from part 6**. On a larger 34-item set of mine, 16K fails the threshold: 0.77 against a 0.825 bar. On NoLiMa's own items with thinking on, the model gets 0.69 at 16K against 0.98 at 1K. Nor does the matching 16,000 entry make Qwen3-8B and GPT-4.1 equivalent: they use different items, and NoLiMa's length grid doubles at each step.

The old absolute band also does **not** describe the measured 2026 frontier. In [[blog/2026-09-27-is-half-your-context-window-still-marketing|Is Half Your Context Window Still Marketing in 2026?]], I measured GPT-6 Luna reaching about **64K with reasoning off and 128K with reasoning high**, out of a 1M window. Its shared-word control stays at 0.93 or above while meaning-based recall falls. The usable distance has moved, and the distinction between fitting text and retrieving through meaning remains.

## Using an LLM at long context

When a task needs a connection across a long document, I would check that connection at the intended length and thinking setting. A short-context quantization result cannot certify long-context recall, and a claimed window cannot certify that a particular question remains answerable throughout it.

For the tasks here, reducing precision to Q4_K_M or Q5_K_M was less damaging than extending the material at BF16. That gives me a reason to test distance explicitly, even when I keep full precision. The useful question is whether the model can still connect the needed facts under the conditions in which I plan to use it.

The length numbers above were re-measured with thinking on; the quantization ladder was not re-run and remains a thinking-off measurement.

*Revised on 2026-09-27: the thinking setting of each experiment stated, a corrected reading of the Q3_K_S failure and the precision threshold, new figures, and a link to the 2026 frontier measurement.*

**References.**

- Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze, "NoLiMa: Long-Context Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167).
- Adobe Research, [NoLiMa GitHub repository](https://github.com/adobe-research/NoLiMa), 2025 model-generation updates.
- Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg, "RULER: What's the Real Context Size of Your Long-Context Language Models?" [arXiv 2404.06654](https://arxiv.org/abs/2404.06654).
- Meta AI, ["Llama 4: Leading Intelligence"](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) launch blog (pretraining sequence length up to 256K, against the 10M claimed window).
- Qwen Team, "Qwen3 Technical Report," [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
