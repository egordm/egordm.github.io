---
title: "Your LLM Read the File. Did It Understand It?"
date: 2026-07-19
draft: false
series: "LLM Comprehension"
series_order: 4
tags:
  - llm
  - interpretability
  - attention
description: "LOCOS finds the heads that push a fact toward an answer, even a wrong one. Ablation and attention knockout held up, but the write score rated clean answers below flawed ones. Finding the source and testing understanding need different tools."
aliases:
  - "logit-contribution-scoring"
  - "synthesis-heads"
---

In [[blog/2026-07-18-which-heads-read-your-context|the previous post]], the small LLM, Qwen3-0.6B, read “The timeout is half a minute.” and answered **“The timeout is half a minute, which is 300 seconds.”** An attribution tool would point at the right sentence. The answer is still wrong.

“Which source did the answer rest on?” and “Did the LLM get that source right?” are different questions. Does looking inside the LLM close that gap? **The tools that find where a fact enters the answer held up; the score that looked like it would say how well did not.** The LOCOS head list survived ablation in the paper and in my reproduction on Qwen3-8B. Attention knockout held up in my tests on Qwen2.5-Coder-7B. But using the LOCOS write score as a meter for understanding gave the wrong result: clean answers scored lower than flawed ones.

## From the read to the push

We met the **matcher**, which gives the relevant fact close attention, and the **mover**, which writes strongly toward the answer. In our example, both can read “half a minute”, but the mover delivers the larger push toward “30” at the **answer position**. That is the position whose vector predicts the next answer token.

LOCOS measures that push. It projects a head's write from source position $j$ onto the answer token's output direction $u_y$:

$$
\phi_j = u_y\cdot\left(\alpha_j W_Ov_j\right).
$$

The attention weight $\alpha_j$ scales the write $W_Ov_j$; the dot product measures how much it points toward the chosen answer token. LOCOS then sums those pushes over the fact and subtracts the push from the remaining context, rescaled to the fact's token length. It is **fact minus rescaled background**, with the question held fixed. The [[blog/2026-07-18-which-heads-read-your-context|previous post]] walks through the read, the write and that subtraction. ([LOCOS paper](https://arxiv.org/abs/2607.01002))

We now have a head list and a score. We can test them separately: does the LLM need these heads, and does a larger push mean a better answer?

## Do the LOCOS heads matter?

The direct test is **ablation**: switch the heads off and see what breaks. The LOCOS paper does this on Qwen3-8B with NoLiMa, where the question shares no words with the fact needed to answer it. The answer score is ROUGE-L: how much of the reference answer the LLM still produces.

With no heads off, the published score is 0.401. Switching off the top 50 LOCOS heads takes it to **0.000**. Switching off the top 50 heads from Wu's copy test leaves 0.292. Arithmetic and plain factual recall stay near their normal level, so these heads are specific to this job. ([LOCOS](https://arxiv.org/abs/2607.01002), [Wu's copy test](https://arxiv.org/abs/2404.15574), [NoLiMa](https://arxiv.org/abs/2502.05167))

I reproduced the comparison before relying on the head list. My scores were 0.412 with no heads off, **0.000** with the LOCOS heads off, and 0.342 with Wu's heads off. Switching off 50 random heads left 0.391. Here are the published results beside mine.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-ablation-dark.svg" alt="Qwen3-8B on NoLiMa. ROUGE-L with no heads off: 0.401 published, 0.412 in my reproduction. Top 50 LOCOS heads off: 0.000 in both. Top 50 Wu heads off: 0.292 and 0.342. Random 50 heads off: 0.391, reproduction only." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-ablation-light.svg" alt="Qwen3-8B on NoLiMa. ROUGE-L with no heads off: 0.401 published, 0.412 in my reproduction. Top 50 LOCOS heads off: 0.000 in both. Top 50 Wu heads off: 0.292 and 0.342. Random 50 heads off: 0.391, reproduction only." />

Qwen3-8B has 1,152 heads. Switching off these 50 is enough to erase the inferred answer on this test. The head list finds machinery the LLM needs. That gives us a reason to trust it for locating the work, but we have not yet tested whether its score measures how well that work was done.

## Can the push measure understanding?

The tempting next step is to turn the push into a quality meter. If the mover carries the fact into the answer, perhaps a larger write toward the right answer means the LLM understood the fact better.

I tested this on **Qwen3-8B**, using NoLiMa-style questions whose answer is a character's name and whose question shares no words with the sentence giving it. At about 15K tokens, 15 questions drew 197 answers. A **clean answer** gave the right name alone. A **flawed answer** gave the right name with several wrong candidates alongside it; 43 of the 197 did this. I scored both kinds on the right name.

If the write score tracked quality, clean minus flawed should be positive. It was **negative**: $-0.035$ over all heads, with a 95% interval from $-0.052$ to $-0.020$, and $-0.093$ over the top 50 heads, with an interval from $-0.165$ to $-0.041$. Both intervals sit entirely below zero. The flawed answers scored higher.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-quality-dark.svg" alt="Clean minus flawed write score on Qwen3-8B. All heads: minus 0.035, 95% interval minus 0.052 to minus 0.020. Top 50 heads: minus 0.093, interval minus 0.165 to minus 0.041. Both intervals are below zero: flawed answers score higher." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-quality-light.svg" alt="Clean minus flawed write score on Qwen3-8B. All heads: minus 0.035, 95% interval minus 0.052 to minus 0.020. Top 50 heads: minus 0.093, interval minus 0.165 to minus 0.041. Both intervals are below zero: flawed answers score higher." />

A shorter run at about 4K tokens pointed the same way, but had only 6 flawed answers over 4 questions, too few to read. The write term itself sits clearly above zero on these answers, so it is not noise. It measures a push. That push does not measure answer quality.

## A strong push can point at the wrong answer

Let us return to “half a minute”. In the previous post, we measured the mover's write along $u_{30}$, the direction for “30”. Now suppose the LLM misreads the fact as a full minute and answers “60”. If we score that answer, we measure along **$u_{60}$**. A head whose write strongly pushes “60” scores high. Nothing in that number says “60” is wrong.

The picture below repeats the write and its projection from the previous post. Each panel chooses the direction of the answer being scored as its horizontal axis. The arrows are schematic: we are comparing what the measurement asks, not assigning new scores.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-direction-dark.svg" alt="The same half-a-minute fact, with schematic writes at the answer position. Scoring 30 projects the write along the output direction for 30. Scoring a mistaken 60 projects along the direction for 60 instead. A strong push can support either token; the projection does not test correctness." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-direction-light.svg" alt="The same half-a-minute fact, with schematic writes at the answer position. Scoring 30 projects the write along the output direction for 30. Scoring a mistaken 60 projects along the direction for 60 instead. A strong push can support either token; the projection does not test correctness." />

We choose which token to measure. The score tells us how strongly the write supports that token; it does not check the token against the meaning of “half a minute”. This also explains what it misses in the quality test: the flawed answers contain the right name, and the heads push it. A score along that name's direction does not judge the extra wrong names. I have not pinned down why flawed answers scored *higher* rather than equal. For a quality meter, the sign alone disqualifies it.

The real **Qwen3-0.6B** replay makes the problem concrete. In the previous post, I supplied the correct answer “30” to compare the detectors. The LLM's own answer was “300”. When I score “300” instead, LOCOS selects the same top head, **layer 20, head 13**, with **0.33**, against **0.434** on “30”. And **9 of its top 10 heads are the same**.

This is one item on a small LLM, an illustration. It shows why a familiar head list and a substantial push are not a verdict on correctness. The same top head supports the supplied right answer and the LLM's wrong answer. LOCOS can help us find that head in either case.

## Does the answer need this fact?

We can still ask a useful causal question: would the answer survive if the answer positions could no longer read the fact? **Attention knockout** blocks attention from those positions to the fact's tokens. We cut the read that feeds the write, then check the answer. The method comes from Geva et al. ([Paper](https://arxiv.org/abs/2304.14767))

I blocked that attention **at every layer**. Blocking only one layer can let the LLM route around the obstruction, the “hydra effect” described by McGrath et al. A surviving answer after that smaller intervention need not mean the fact was unnecessary. ([Paper](https://arxiv.org/abs/2307.15771))

On **Qwen2.5-Coder-7B**, I used planted facts the answers depend on. Blocking the read at every layer dropped correctness to **0.00**, for both the copy version and the version needing inference of each question. A separate sweep blocking five layers at a time placed the loss in **layers 19 to 27**. Below, our timeout example illustrates the blocked read; the correctness result is from the planted-fact tests.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-knockout-dark.svg" alt="Attention from the answer position to the half-a-minute fact is crossed out throughout a schematic layer stack. In the Qwen2.5-Coder-7B experiment, blocking every layer dropped correctness to 0.00 for both copy and inference versions. A separate sweep blocking five layers at a time placed the loss in layers 19 to 27." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-knockout-light.svg" alt="Attention from the answer position to the half-a-minute fact is crossed out throughout a schematic layer stack. In the Qwen2.5-Coder-7B experiment, blocking every layer dropped correctness to 0.00 for both copy and inference versions. A separate sweep blocking five layers at a time placed the loss in layers 19 to 27." />

Knockout answers “Does the answer need this read?” In these tests, yes. It does not answer “Did the LLM get the fact right?” Cutting a path can show that the answer depends on it without judging what travelled along it. This is evidence about where, not how well.

> [!note]- Where these tools break
> - **The active heads change between examples.** A head list computed once overlaps the heads active on a given example by a Jaccard index of only 0.18 to 0.46. A fixed list is an approximation. ([arXiv 2602.11162](https://arxiv.org/abs/2602.11162))
> - **Positions blur with depth.** A later position's vector mixes information from many tokens. Reading position $j$ does not mean reading only the original token at $j$. The claim that survives is at the circuit level: these heads are needed for this behaviour.
> - **Related background can lower the score.** LOCOS's own caveat is that background on the same topic can penalize heads doing legitimate broad matching when its push is subtracted from the fact's.

## So, did it understand?

We can now put each tool beside the question it answers. In the [[blog/2026-07-12-did-your-agent-actually-read-that-file|first post]], ContextCite ranked the planted source first in **55 of 55 runs**. Here, ablation supported the LOCOS head list, and knockout showed that the answers needed the read. The proposed quality meter failed.

<img class="theme-dark-only figure-center" src="/blog/assets/comprehension-v-toolbox-dark.svg" alt="ContextCite: can we locate the source the answer rests on? Yes. LOCOS head list: can we locate heads needed to carry the fact? Yes. Attention knockout: can we test whether the answer needs the read? Yes. LOCOS write score: can we judge answer quality? No." />
<img class="theme-light-only figure-center" src="/blog/assets/comprehension-v-toolbox-light.svg" alt="ContextCite: can we locate the source the answer rests on? Yes. LOCOS head list: can we locate heads needed to carry the fact? Yes. Attention knockout: can we test whether the answer needs the read? Yes. LOCOS write score: can we judge answer quality? No." />

For the timeout example, finding the sentence, the mover and the push still leaves us with “300”. To test understanding, we have to check what the LLM can do with the fact: ask questions that require it to connect information rather than simply copy words, and judge the answers.

**The test of understanding is behavioural.** Attribution and the internal tools tell us where to look; the answer tells us whether the LLM used the fact correctly. That is the test the rest of this series builds on, from [[blog/2026-07-22-does-a-reasoning-model-actually-read-its-own-thinking-trace|an LLM's own reasoning trace]] to [[blog/2026-07-24-is-half-your-context-window-just-marketing|how much of the context window it can still use]].

---

*Revised on 2026-09-26: retitled from "Your Agent Read the File. Did It Understand It?" and rewritten as the sequel to [[blog/2026-07-18-which-heads-read-your-context|Which Heads Read Your Context?]], with new figures.*

**References.**

- Gema, Alex, Minervini, "Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads" (LOCOS), [arXiv 2607.01002](https://arxiv.org/abs/2607.01002); code at [github.com/aryopg/locos](https://github.com/aryopg/locos).
- Wu, Wang, Xiao, Peng, Fu, "Retrieval Head Mechanistically Explains Long-Context Factuality," [arXiv 2404.15574](https://arxiv.org/abs/2404.15574).
- Modarressi et al., "NoLiMa: Long-Context Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167).
- Geva, Bastings, Filippova, Globerson, "Dissecting Recall of Factual Associations in Auto-Regressive Language Models," EMNLP 2023, [arXiv 2304.14767](https://arxiv.org/abs/2304.14767).
- McGrath, Rahtz, Kramár, Mikulik, Legg, "The Hydra Effect: Emergent Self-repair in Language Model Computations," [arXiv 2307.15771](https://arxiv.org/abs/2307.15771).
- On the per-example churn of the active head set: [arXiv 2602.11162](https://arxiv.org/abs/2602.11162).
