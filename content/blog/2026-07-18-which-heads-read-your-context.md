---
title: "Which Heads Read Your Context?"
date: 2026-07-18
draft: false
series: "LLM Comprehension"
series_order: 3
tags:
  - llm
  - interpretability
  - attention
description: "Which attention heads carry a fact into an answer? We work through Wu's copy test, QRHead and LOCOS on the same example, then see why they select different heads in a real LLM."
aliases:
  - "retrieval-head-detectors"
---

**Which heads carry a fact from the context into the answer?** We have asked [[blog/2026-07-12-did-your-agent-actually-read-that-file|which source an answer rested on]] and [[blog/2026-07-17-how-attention-works|how attention moves information]]. Let us now follow a fact through the heads themselves.

> Logs rotate daily. **The timeout is half a minute.** Retries use backoff.
>
> **Question:** How many seconds is the timeout?
>
> **Answer:** 30

The answer is absent from the context. A head could find the right sentence without helping convert “half a minute” into “30”. We need the distinction from the attention post: a head **reads** by assigning attention weights to source tokens, then **writes** by transforming their value vectors, scaling them by those weights, and adding the result to the residual stream, the vector carried between layers.

## Where do we measure?

We will use three detectors: Wu's copy test, QRHead and LOCOS. **QRHead reads from the question's tokens**, averaging their attention to the fact. **Wu and LOCOS measure at the step that writes an answer token.** The empty slot below marks this **answer position**, which the model fills with “30”. We inspect the state that predicts that token.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-positions-dark.svg" alt="The example as a sequence of schematic token boxes: Logs rotate daily. The timeout is half a minute. Retries use backoff. How many seconds is the timeout? An empty answer slot follows, which the model fills with 30. QRHead measures attention from the highlighted question tokens; Wu and LOCOS measure at the step that fills the answer slot." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-positions-light.svg" alt="The example as a sequence of schematic token boxes: Logs rotate daily. The timeout is half a minute. Retries use backoff. How many seconds is the timeout? An empty answer slot follows, which the model fills with 30. QRHead measures attention from the highlighted question tokens; Wu and LOCOS measure at the step that fills the answer slot." />

Attention is over **tokens**. Whenever we summarize a sentence or another part of the input, its attention is the **sum over its tokens**.

**For this invented example only, I use the same attention pattern at the answer step and averaged over the question's tokens.** The real LLM later in the post will show why we must distinguish them.

## Meet the three heads

I gave our invented heads plain roles inspired by real findings, though many real heads have no single clean role:

- **Head A, the matcher,** looks for the sentence that best matches the question. It pays close attention to our timeout fact.
- **Head B, the mover,** carries answer-relevant content into the output. It gives the fact less attention than A, but writes strongly toward “30”. The name echoes Wang et al.'s *name-mover heads*, which move a name from the context toward the output; B's conversion is my illustration. ([Paper](https://arxiv.org/abs/2211.00593))
- **Head C, the resting head,** parks most of its attention on the first token whatever the question. This illustrates an *attention sink*, the tendency to attend to initial tokens even when they have little semantic importance, described by Xiao et al. ([Paper](https://arxiv.org/abs/2309.17453))

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-heads-dark.svg" alt="Three invented roles. A, the matcher, gives the fact close attention but delivers a small push toward 30. B, the mover, gives it less attention but delivers a strong push toward 30. C, the resting head, parks attention on the first token regardless of the question." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-heads-light.svg" alt="Three invented roles. A, the matcher, gives the fact close attention but delivers a small push toward 30. B, the mover, gives it less attention but delivers a strong push toward 30. C, the resting head, parks attention on the first token regardless of the question." />

We can organize the detectors around one recipe: give source tokens credit, sum over the labelled fact, subtract a baseline, then rank heads. Wu's copy test is the parent, with no baseline subtraction; QRHead and LOCOS change the credit and baseline.

## Wu: does the head read the token being emitted?

At the step that emits “30”, we find a head's **single most-attended source position**. Wu's test gives credit only if that position holds the emitted token and lies inside the planted fact. Over an answer, the score is the fraction of answer tokens that pass this test. This is the copy criterion introduced by Wu, Wang, Xiao, Peng and Fu. ([Paper](https://arxiv.org/abs/2404.15574))

In our example, “30” appears nowhere in the context. No head can pass, however much attention it gives “half a minute”. To see the test succeed, let us use the **copy version**: “The timeout is 30 seconds.” I specify one extra detail for Head A: its most-attended source token is now “30”. We can compare that token directly with the emitted one.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-wu-dark.svg" alt="Wu's token comparison. In the half-a-minute fact there is no source token 30, so every head scores 0. In the copy version, Head A's strongest attention points to the source token 30 inside the fact; it matches the emitted token 30, so A scores 1." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-wu-light.svg" alt="Wu's token comparison. In the half-a-minute fact there is no source token 30, so every head scores 0. In the copy version, Head A's strongest attention points to the source token 30 inside the fact; it matches the emitted token 30, so A scores 1." />

**Wu selects no head on “half a minute”; A scores $1$ in the copy version.** The test detects exactly the operation it was designed to find. Notice why we needed the extra token-level detail: the fact's total attention cannot tell us which individual token had the largest weight. Wu's paper finds a small, stable set of heads with this copying behaviour.

## QRHead: does the question draw attention to the fact?

Let us return to “half a minute”. QRHead, from Zhang, Yin, Yen, Chen and Ye, gives each source token credit for the attention it receives **from the question's tokens, averaged over them**. We sum that credit over the fact, then repeat the measurement with the question replaced by “N/A”. The context stays fixed. ([Paper](https://arxiv.org/abs/2506.09944))

For Head A, the fact receives $0.70$ with the real question and $0.10$ with “N/A”. Subtracting leaves $0.60$: attention to this fact above the head's interest without the question.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-qrhead-dark.svg" alt="QRHead compares fact attention with the real question and with N/A, averaged over the question tokens and summed over fact tokens. A: 0.70 minus 0.10 equals 0.60. B: 0.30 minus 0.15 equals 0.15. C: 0.05 minus 0.05 equals 0.00. A has the largest increase." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-qrhead-light.svg" alt="QRHead compares fact attention with the real question and with N/A, averaged over the question tokens and summed over fact tokens. A: 0.70 minus 0.10 equals 0.60. B: 0.30 minus 0.15 equals 0.15. C: 0.05 minus 0.05 equals 0.00. A has the largest increase." />

**QRHead picks A.** B's increase is smaller, and C's attention to the fact does not change. C also shows why the baseline matters: it gives the start $0.80$ whether we ask the question or supply “N/A”. If the fact sat there, that large weight could look like retrieval. The subtraction, $0.80-0.80=0$, exposes the habit.

Writing $F$ for the fact's token positions and $\bar\alpha_j$ for attention to source token $j$ averaged over the question's tokens, we have

$$
S_{\mathrm{QR}} = \sum_{j\in F}\bar\alpha_j^{\mathrm{question}}
- \sum_{j\in F}\bar\alpha_j^{\mathrm{N/A}}.
$$

The question is what changes between the two runs. We never need to inspect an answer. But we still do not know what the head writes from the fact. That is where LOCOS begins.

## LOCOS: does the write push toward the answer?

LOCOS, from Gema, Alex and Minervini, measures **at the step that writes each answer token**. We keep the question fixed and follow the information coming from a source position $j$ inside “half a minute”. ([Paper](https://arxiv.org/abs/2607.01002))

First, $v_j$ is the head's packaged content from that position. Applying the output matrix gives $W_Ov_j$, the full-strength write if the head attended entirely to that position. Multiplying by its actual attention weight gives $\alpha_j W_Ov_j$, the write actually delivered to the residual stream.

Now we need to ask how much of that write points toward “30”. Think of forces acting on an object: we choose an axis, project each force onto it, and add the signed projections to find the net push along that axis. Here the axis is “toward the answer token”, and its direction comes from $u_y$, that token's vector in the model's output, or **unembedding**, matrix. The resulting credit is

$$
\phi_j = u_y\cdot\left(\alpha_j W_Ov_j\right).
$$

Attention scales a full-strength write into the write actually delivered. Its projection onto the answer direction tells us how much it helps “30”.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-locos-write-dark.svg" alt="LOCOS builds a source token's credit from its value v_j, through the full-strength write W_O v_j, the attention-scaled write alpha_j W_O v_j, and the projection phi_j along u_30. Beside this chain, equal-scale vector plots show A's write mostly across the answer direction and B's write mostly along it. Their weighted projections from the fact are 0.07 and 0.90." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-locos-write-light.svg" alt="LOCOS builds a source token's credit from its value v_j, through the full-strength write W_O v_j, the attention-scaled write alpha_j W_O v_j, and the projection phi_j along u_30. Beside this chain, equal-scale vector plots show A's write mostly across the answer direction and B's write mostly along it. Their weighted projections from the fact are 0.07 and 0.90." />

A positive projection helps the answer token's score; a negative one opposes it. The projections sum to the net direct push, in logit units. **The answer token need not appear in the input**: $u_{30}$ comes from the output side of the model, so we can project onto it even when the context contains only “half a minute”.

For the invented vectors, I use coordinates “toward 30” and “other”. A's full-strength write from the fact is $(0.1, 1.0)$, B's is $(3.0, 0.2)$, and C's is $(0.2, 0.1)$. Multiplying the first coordinate by the fact's attention gives A $0.70\times0.1=0.07$, B $0.30\times3.0=0.90$, and C $0.05\times0.2=0.01$. B reads the fact less than A, yet delivers the larger push toward the answer.

> [!note]- How the illustrated vectors relate to the model
> Each illustrated full-strength write from the fact is an attention-weighted average of its tokens' $W_Ov_j$ vectors. Multiplying by the fact's total attention recovers their summed contribution. I draw $u_{30}$ as a unit vector along the horizontal axis; in the model, the dot product also includes the unembedding vector's scale.
>
> We choose the answer token's output direction as our axis. This differs from PCA, where the axes come from variation in the data.

### Why subtract the rest of the context?

We have measured the push from the fact, but it might be part of a push coming from everywhere. LOCOS asks whether it **concentrates on the fact**. We sum $\phi_j$ over the fact, then subtract the sum over the remaining context tokens, rescaled to the fact's token length. If $R$ contains those remaining positions, the score is

$$
S_{\mathrm{LOCOS}} = \sum_{j\in F}\phi_j
- \frac{|F|}{|R|}\sum_{j\in R}\phi_j.
$$

The length adjustment gives us a fair comparison: the fact's push minus the background push we would expect over the same number of tokens. The question stays fixed and is not part of the context baseline.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-locos-contrast-dark.svg" alt="The timeout fact feeds the sum of phi over fact tokens. The surrounding logs and retries sentences feed the sum over the rest of the context, rescaled by the fact length divided by the rest length. Subtracting background from fact gives the LOCOS score. With near-zero background in the example, A scores 0.07, B 0.90, and C 0.01; B wins." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-locos-contrast-light.svg" alt="The timeout fact feeds the sum of phi over fact tokens. The surrounding logs and retries sentences feed the sum over the rest of the context, rescaled by the fact length divided by the rest length. Subtracting background from fact gives the LOCOS score. With near-zero background in the example, A scores 0.07, B 0.90, and C 0.01; B wins." />

**LOCOS picks B.** I made the example's background near zero, so the subtraction leaves the fact pushes effectively unchanged. The contrast matters when a head also writes toward the answer from other context positions. If the model answers from memory without needing the context, and no push concentrates on the fact, the contrast stays near zero. That does not by itself establish where the answer came from.

> [!note]- What this score leaves out
> LOCOS measures a direct-path attribution. Final normalization sits between the residual stream and the logits, and later heads and MLPs can respond to earlier contributions. The raw projection is not an exact change in answer probability, a causal effect of removing the head, or evidence that the answer is correct. LOCOS also reports an attention-only control: it keeps the fact-versus-background comparison but credits attention instead of the write.

The same “half a minute” fact gives us different verdicts because each test rewards a different behaviour: literal copying, question-driven attention, or a direct push toward “30”.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-verdicts-dark.svg" alt="On the half-a-minute example, Wu selects no head because there is no 30 to copy. QRHead selects A, the matcher. LOCOS selects B, the mover. In the copy version, Wu detects A too." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-verdicts-light.svg" alt="On the half-a-minute example, Wu selects no head because there is no 30 to copy. QRHead selects A, the matcher. LOCOS selects B, the mover. In the copy version, Wu detects A too." />

## A real LLM separates them too

I ran the three tests on **Qwen3-0.6B**, with 28 layers and 16 heads per layer, 448 heads in total. The script ran QRHead and LOCOS implementations plus a few lines for Wu's test. The context contains seven short sentences of notes, with “The timeout is half a minute.” in the middle. The question is “What is the timeout in seconds? Reply with a single number.”[^walkthrough]

On its own, the LLM answers **“The timeout is half a minute, which is 300 seconds.”** On the copy version, it answers “30”. The first answer is wrong. To compare the detectors on the same answer, I fed the correct answer “30” to both versions. The following scores describe that supplied answer, not the LLM's freely generated wrong answer. The script averages the answer-step measurements over the supplied answer's tokens.

Wu finds **0 of 448** heads above zero on the half-a-minute item. QRHead's top head is **layer 13, head 4**, with score **0.151**. LOCOS gives that head **0.010**. LOCOS's top head is **layer 20, head 13**, with score **0.434**; QRHead gives it **0.0045**.

Let us separate the two moments here. At the answer step, layer 13, head 4 gives the fact **0.034** attention, while layer 20, head 13 gives it **0.315**. QRHead's measurement during the question does not stand in for attention while writing the answer. The QRHead score subtracts a baseline; the answer-step attention here is raw. Both attention measurements sum over the fact's tokens.

<img class="theme-dark-only figure-center" src="/blog/assets/heads-v-real-dark.svg" alt="Qwen3-0.6B on the half-a-minute item with answer 30 supplied. Layer 13 head 4: QRHead 0.151, fact attention at answer step 0.034, LOCOS 0.010. Layer 20 head 13: QRHead 0.0045, fact attention at answer step 0.315, LOCOS 0.434. Top-10 lists share 0 heads; rank correlation over 448 heads is minus 0.04." />
<img class="theme-light-only figure-center" src="/blog/assets/heads-v-real-light.svg" alt="Qwen3-0.6B on the half-a-minute item with answer 30 supplied. Layer 13 head 4: QRHead 0.151, fact attention at answer step 0.034, LOCOS 0.010. Layer 20 head 13: QRHead 0.0045, fact attention at answer step 0.315, LOCOS 0.434. Top-10 lists share 0 heads; rank correlation over 448 heads is minus 0.04." />

Across all heads, QRHead's and LOCOS's top-10 lists share **0 heads**, and their rank correlation is **-0.04**.

The copy version brings Wu back: it finds **8 heads**, **5** of them in LOCOS's top 10 and **none** in QRHead's top 10. Between versions, QRHead keeps **8 of its top 10** heads; LOCOS keeps **1**. QRHead never looks at the answer. LOCOS follows the write toward it, and on this item, copying “30” and converting “half a minute” run through different heads. In the invented example, Wu's copy head was also QRHead's choice. The real item shows that this agreement is not a rule.

This is **one item on a small model: an illustration, not a measurement** of how often detectors agree. The published comparisons also find different sets: only **8 of QRHead's top 32** overlap Wu's, and **2 of LOCOS's top 10** overlap Wu's on Qwen3-8B. LOCOS reports Wu's top head scoring **0.97** on a literal needle test and **0.03** on [NoLiMa](https://arxiv.org/abs/2502.05167), where questions share no words with the fact they need. ([QRHead](https://arxiv.org/abs/2506.09944), [LOCOS](https://arxiv.org/abs/2607.01002))

## Which detector I would use

I would use **Wu's test for literal copying**. Its criterion is clear: the strongest attention points to the same token inside the fact. A zero on an inferred answer tells me that the copy criterion did not fire.

I would use **QRHead to find question-relevant sources**. With labelled spans, it selects heads whose attention responds to the question. Its authors keep 16 heads for models under 10B, then use those heads as a retriever, improving multi-hop question answering and re-ranking. It can rank sources without inspecting an answer. ([Paper](https://arxiv.org/abs/2506.09944), [code](https://github.com/princeton-pli/QRHead))

I would use **LOCOS to trace the direct push from a fact toward a given answer**, including an answer absent from the source text. It combines the read with the write and compares the fact with the rest of the context. ([Paper](https://arxiv.org/abs/2607.01002), [code](https://github.com/aryopg/locos))

We now have a precise meaning for “this head carries the fact”: copied token, question-driven attention, or direct push toward the answer. But the small LLM also produced “300 seconds”. Finding a path from the fact to an answer leaves a further question: did the LLM use that fact correctly? That is where [[blog/2026-07-19-your-agent-read-the-file-but-did-it-understand-it|Your LLM Read the File. Did It Understand It?]] begins.

[^walkthrough]: The Qwen3-0.6B numbers come from one scripted run of `heads_walkthrough.py`, beside the figure sources, recorded in `heads_walkthrough.json`. The three-head example is invented to make the arithmetic visible. “30” is treated as one answer token in that example; the script uses the model's actual tokenizer and averages over the supplied answer's tokens.

**References.**

- Wu, Wang, Xiao, Peng, Fu, "Retrieval Head Mechanistically Explains Long-Context Factuality," [arXiv 2404.15574](https://arxiv.org/abs/2404.15574).
- Zhang, Yin, Yen, Chen, Ye, "Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking," EMNLP 2025, [arXiv 2506.09944](https://arxiv.org/abs/2506.09944); [code](https://github.com/princeton-pli/QRHead).
- Gema, Alex, Minervini, "Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads," [arXiv 2607.01002](https://arxiv.org/abs/2607.01002); [code](https://github.com/aryopg/locos).
- Modarressi et al., "NoLiMa: Long-Context Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167).
- Elhage et al., "A Mathematical Framework for Transformer Circuits," [Anthropic, 2021](https://transformer-circuits.pub/2021/framework/index.html).
- Wang, Variengien, Conmy, Shlegeris, Steinhardt, "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small," [arXiv 2211.00593](https://arxiv.org/abs/2211.00593).
- Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks," [arXiv 2309.17453](https://arxiv.org/abs/2309.17453).
