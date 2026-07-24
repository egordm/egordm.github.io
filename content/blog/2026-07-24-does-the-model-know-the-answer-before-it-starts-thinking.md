---
title: "Does the Model Know the Answer Before It Starts Thinking?"
date: 2026-07-24
draft: true
tags:
  - llm
  - agents
  - interpretability
  - reasoning
description: "We trained the first tuned lens for a Qwen model and asked it, at the last token before generation starts, whether a planted answer is already decodable deep in the network. Every cell came back silent. The reason turns out to be more interesting than a miss: a thinking model's truthful next word is never the answer, it's the opening line of its own reasoning trace."
aliases:
  - "tuned-lens-timing"
  - "prompt-end-decodability"
---

[[blog/2026-07-24-is-half-your-context-window-just-marketing|The previous post]] measured whether a planted fact survives long enough in context to still be reachable, and found the model's reliable range lands at roughly half its advertised window. That post, and the two before it, all grade the same kind of thing: a finished output. Did the answer lean on the right source, did it read that source correctly, did it still have the fact once enough tokens had piled up. All three questions wait for generation to finish before asking anything.

This post asks a question none of the first four can answer by construction: at what depth inside the network does an answer become available, before a single token of the response exists? If you could watch an answer's rank climb toward the top of the model's vocabulary by, say, layer 20 of 36, before the model has written "let me think" or anything else, you could catch a retrieval miss before it burns through the model's full thinking budget, up to 4,096 tokens on the run in this series, chasing a fact that was never going to surface.

We built the instrument to check. It came back with a clean zero, and the story of why is the most useful result in this series so far.

## A tuned lens, trained for a Qwen model for the first time

A transformer's last layer turns its final internal state into a probability over the vocabulary. Every layer *before* the last one has its own internal state too, in the same vector space, and a **lens** is a small tool that asks an earlier layer's state the same question the final layer answers: if you had to guess the next token right now, what would you say?

A raw ("logit") lens just runs the model's own final read-out machinery on an earlier layer's state as-is. That is cheap but biased: earlier layers write to the residual stream in a different basis than the one the final read-out expects, so a raw lens reads mostly noise. The **tuned lens** (Belrose et al., [arXiv 2303.08112](https://arxiv.org/abs/2303.08112)) fixes this with one small learned affine translation per layer. Once trained, a translator lets you ask, at any layer, "what would the model's output head say if it had to answer right now, using only what's arrived at this depth?"

![[blog/assets/tuned-lens-translator-schematic.png|340]]
*The tuned lens: take an intermediate layer's hidden state, apply that layer's learned translation, and read the result out through the model's own unembedding, skipping the layers above. Figure 2 of Belrose et al., ["Eliciting Latent Predictions from Transformers with the Tuned Lens"](https://arxiv.org/abs/2303.08112), arXiv 2303.08112 (CC BY 4.0).*

The training objective is the part everything later turns on. The translator at layer $l$ is one affine map $(A_l, b_l)$, read out through $W_U$, the model's own output head, and trained to make that read-out match the model's final next-token distribution:

$$
q_l(x_{t+1}) = \mathrm{softmax}\big(W_U (A_l h_l + b_l)\big), \qquad \min_{A_l,\, b_l} \ \mathrm{KL}\big(p_{\mathrm{final}} \,\|\, q_l\big)
$$

In words: $q_l$ is what layer $l$'s state would say if forced through the model's own unembedding right now, and training pulls $q_l$ toward $p_{\mathrm{final}}$, the distribution the full model actually emits, so a tuned lens is calibrated to predict the model's next emission, not to report everything a layer knows.

No public tuned lens existed for any Qwen model when we started. The released training and inference code doesn't even support the architecture, a hardcoded check blocks it before you get the chance to try. We patched the architecture check (Qwen's internal layer structure is identical to the already-supported Llama family) and trained our own translator set on Qwen3-8B, thinking mode on: 250 steps, KL self-distillation against the model's own final-layer logits, one RTX 4090, running overnight, about 13 hours.

We trained only 19 of the model's 36 layers, the middle band where a lens has a chance of saying anything useful (the earliest layers are known to be unreliable across model families, and the latest layers necessarily collapse toward the final answer by construction, since that's what the translator is trained to match). Training the other 17 layers would have cost roughly a full extra day of GPU time for reads nobody planned to trust.

## The calibration bar the lens has to clear first

A lens is a measuring instrument, and an instrument you haven't calibrated is a rumor. Before reading a single trajectory, every trained layer's translator had to clear a calibration bar: how close does its output distribution sit to the model's own real final-layer distribution, in bits of KL divergence, on held-out text the model never trained on?

The relative bar passed everywhere: all 19 trained layers beat the raw, untuned lens by roughly a factor of four. The absolute bar, a fixed ceiling on how much divergence is still trustworthy, split the band cleanly in two:

| layer range | KL divergence (bits) | absolute ceiling (4.0 bits) |
|---|---|---|
| layers 9-16 (shallow) | 4.02 to 4.77 | fails |
| layers 17-27 (deep) | 2.20 to 3.99 | passes |

The pass/fail split is monotonic with depth (calibration improves layer by layer), and the boundary lands exactly between layer 16 and layer 17. We do not read a scale where it doesn't calibrate: every result below is scoped to the licensed band, layers 17 through 27, 11 layers spanning roughly the back half of the network's depth. The shallow layers stay trained and reportable but never gate a verdict.

## The setup: read the last token before the model says anything

The material is the same 14 needles from the previous post's frozen set, reused unchanged: facts bound to invented entities, with zero lexical overlap between the question and the planted sentence, so a frequency prior or a keyword match can't produce the answer at any depth. Each needle runs at two haystack lengths, 1,000 and 4,000 tokens, in two conditions: **present** (the real haystack) and **absent** (the identical haystack with the needle sentence swapped for a same-pool filler, otherwise byte-identical). Fourteen needles times two lengths times two conditions is 56 cells, 28 present and 28 absent.

The read position is the last token of the rendered prompt, the instant after the model has seen everything and before it has produced anything. This position exists identically whether the model is about to answer correctly or not, so it can never be an ill-posed control the way a position chosen for *where the answer gets said* would be. A layer is scored as **decodable** when the answer token ranks in the top 10 of that layer's translator output, sustained for three consecutive layers in the band (a one-layer blip never counts).

![[blog/assets/timing-setup-diagram.png|700]]
*The two read positions: prompt-end (blue), the instant before any output exists, and first-emission (green), the token where the answer starts being said; at each, the tuned lens reads every layer in the stack above it, and only the calibrated band (layers 17-27) gates a verdict.*

Two bars were fixed before a single trajectory was read: the present condition had to onset somewhere in the band on at least 85 percent of cells, and the absent condition had to stay silent on at least 90 percent of cells. Miss either, and the lens does not validate as a measure of whether the answer is available at that position.

There is a real base-rate hazard this design has to defeat. Roughly 60 to 80 percent of a model's early top-ranked guesses get overwritten by the final layer in normal operation ("guess-then-refine," "How Do LLMs Use Their Depth?", [arXiv 2510.18871](https://arxiv.org/abs/2510.18871)), so a bare mid-layer-correct-then-overwritten pattern proves nothing on its own. The invented-entity construction closes that door: nothing but the planted sentence can produce the right answer at any layer, so any present-versus-absent gap is attributable to the plant, not to background noise about how transformers normally revise their guesses.

## The result: zero

Present onset fraction: 0.0, zero of 28 cells. Absent onset-free fraction: 1.0, 28 of 28. The split is identical at both lengths, zero of 14 at 1,000 tokens and zero of 14 at 4,000 tokens. Not one of the 56 cells showed even a single-layer rank blip inside the licensed band; the best rank recorded anywhere in the band, present or absent, is 11, one place outside the top-10 decodability threshold.

One representative cell makes the shape concrete. Here is the plant answer's rank at the last context token, present condition, tracked across the licensed band:

| layer | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rank | 300 | 289 | 224 | 215 | 114 | 95 | 1,507 | 21,794 | 18,652 | 23,192 | 40,952 |

![[blog/assets/timing-rank-trajectory.png|700]]
*One representative cell (Qwen3-8B, thinking on): the prompt-end read worsens with depth, from its best rank of 95 mid-band to 40,952 at layer 27, while the first-emission read of the same cell dives under the top-10 decodability bar to rank 2.*

The rank climbs into the mid-band, briefly touches its best point around layer 22, then gets *worse* the deeper you go. That is not the shape a genuinely emerging answer produces. It is the shape of something else entirely.

## Why zero: the lens is answering a different question, truthfully

A tuned lens is trained to predict one thing: what would the model's output head say right now, given only what has arrived at this layer. At the last token of the prompt, before any generation, that question has an honest answer, and it is not "the fact I was asked about." It is "the first word of my reasoning trace." A model running in thinking mode does not open with its answer. It opens with a think-opener, a restatement, a "let me work through this," and the plant fact, wherever it sits in context, has not been pulled to the surface yet.

Picture the lens as an interviewer standing at every layer, asking the same question over and over: "what will you say next?" A well-trained interviewer gets a truthful answer. The better calibrated a layer is, meaning the more faithfully its translator reproduces what the model's own head would actually say, the more truthfully it reports that the model is about to start narrating, not answering. That is exactly the shape in the table above: rank *worsens* through the deep layers, because the deeper, better-calibrated layers are the ones most confidently reporting the true next word, and the true next word is never the fact.

The zero is not 28 failed retrievals. It is 28 truthful "let me think"s, read by an instrument that was asking the right question of the wrong position.

## Proof the instrument works: read where the model actually answers

If that account is right, the same lens, at the same layers, should read the answer cleanly at the one position where the truthful next word genuinely *is* the answer: the position immediately before the model emits it, deep inside its own trace or at the final response.

It does. Read there, 11 of the 28 present-condition cells show the answer onsetting in the deep band, concentrated at layers 24 and 25, with the answer sitting at rank 2 or 3 by the last few layers. In the same representative cell from above, the emission-position trajectory at layers 24 through 27 reads 181, 3, 2, 2. The rank collapses to near-certain the instant you move the read position to where the model is genuinely about to say the fact out loud.

The lens works. The translator, the calibration, the capture pipeline, all of it decodes a planted answer cleanly when asked at the position where the model's truthful next word is that answer. The zero at prompt-end is not an instrument failure; it's the instrument correctly reporting that, at that specific position, the truthful next word is something else.

## What a zero cannot say

A verdict this clean still has a hard limit, and it is worth stating exactly rather than rounding it away. "The answer is not there yet" and "the answer is there, but invisible to a reader that can only see what the model is about to say" are indistinguishable in this data. The lens is built entirely out of the model's own next-token machinery; asking it a next-token question at a position whose true next token is a think-opener will always read this way, whether or not the fact has, in some other sense, already arrived at that position. This is exactly why the result is written up as a verdict on the *instrument*, not a claim about the model's internal state: it says the prompt-end reading of this lens does not validly measure availability in the thinking regime, not that the answer wasn't there.

A second, independent check closes off the most tempting escape hatch: maybe the top-10 cutoff is just too strict, and a weaker signal is sitting just below it. It isn't. Looking at each cell's single best rank anywhere in the band, not just whether it crossed the top-10 line, the present condition's median best rank is 82.5. The absent condition's median best rank, where there is no fact to find, is 75.5. Present is not ranked better than absent. Nothing is quietly excluded by the cutoff; there is no gradient of a hidden signal building toward the surface, at least not one this class of instrument can see. Running the identical comparison through the untuned raw lens instead produces the same shape at wildly different scale (median best rank around 32,000 present versus 22,000 absent), confirming this isn't an artifact of the trained translator specifically: no decoder built this way, tuned or raw, out-ranks present against absent at this position.

The timeline deserves one honest sentence too: a two-cell smoke run had already shown the emission-position lock before the full grid ran, and the pre-registered prompt-end read ran anyway, because the alternative, swapping the statistic after peeking at the data, is how measurement stops meaning anything.

Where that leaves the question, stated plainly: this asking of it is closed, with a mechanism. Decodable in the next-emission distribution, before generation starts, in a thinking model: no, and the reason is now a named property of the read-out rather than a mystery. The question itself is not closed. A reader that does not route through the next-emission distribution, or a read position where the reasoning has finished but the answer is not yet spoken, are both constructible in principle, and this data says nothing about what either would see.

## The bonus finding: nothing gets lost after retrieval

One more question rides along for free, because the same instrument that reads a clean present-versus-absent split can also be pointed at the cells the model got *wrong*. The "knew it, then lost it" story, an answer that was momentarily correct mid-network and got overwritten before the final output, is a real and previously documented failure mode in other models. Does it happen here?

Across 24 wrong-outcome cells at the two longest tested haystack lengths (7 at 16,000 tokens, 17 at 24,000), reading the same present-versus-absent, last-token-before-generation setup: present onset is zero of 24, absent stays silent at 24 of 24, and the count of cells where the answer was ever accessed but the final answer came out wrong anyway is exactly zero. Every deep failure in this population is a clean retrieval miss: the fact never surfaced at all, not a fact that surfaced and then got overwritten. That corroborates the previous post's finding directly. What degrades with distance is the hop into the fact, not something that happens to it after the hop succeeds.

## What it means for builders

**Pre-registering the pass and fail bars before running the grid is what turned a zero into a verdict instead of an excuse.** A zero read after the fact invites a story fitted to the number. A zero read against bars fixed in advance, with a positive control built into the same design that proves the instrument can read a clean signal when one exists, is a finding you can act on.

**Instrument validity has to be checked before instrument readings get trusted, every time, not once.** The calibration bar cut the usable depth range in half before a single decodability number existed. The positive control, reading the emission position with the identical lens, is what separates "this instrument is broken" from "this instrument answered a different, real question." Skip either step and a clean zero looks like either a bug or a discovery; it's neither.

**A negative with a mechanism is a result, not a dead end.** "The lens reads zero at this position" is not useful on its own. "The lens reads zero at this position because a thinking model's truthful next word there is never the answer" tells you exactly where a timing read would need to move to say something about availability: closer to where the model actually commits to answering, not at the boundary between context and generation.

The trained translator set and its calibration record are candidates for release once they're packaged properly. No link and no date yet, stated here rather than promised.

## The shape of a truthful no

We asked whether the answer is already there, deep in the network, before the model starts thinking. The instrument said no, cleanly and consistently, at every one of 56 cells. That no turned out to be the correct answer to a slightly different question than the one we thought we were asking: not "is the fact available," but "is the fact what the model is about to say next," and for a model trained to narrate before it answers, those are not the same question at the position where context ends and generation begins.

The same lens, moved one position later, answers the original question fine. That's the part worth sitting with. An instrument that fails cleanly, with a mechanism you can name and a control that proves it isn't simply broken, has told you something true about where the measurement has to be taken, not just what it currently reads.

---

**References.**

- Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman, Steinhardt, "Eliciting Latent Predictions from Transformers with the Tuned Lens," [arXiv 2303.08112](https://arxiv.org/abs/2303.08112).
- "How Do LLMs Use Their Depth?", [arXiv 2510.18871](https://arxiv.org/abs/2510.18871).
- Qwen Team, "Qwen3 Technical Report," [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
