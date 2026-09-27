---
title: "Does a Reasoning Model Actually Read Its Own Thinking Trace?"
date: 2026-07-22
draft: false
series: "LLM Comprehension"
series_order: 5
tags:
  - llm
  - agents
  - interpretability
  - reasoning
  - chain-of-thought
description: "I cut, corrupt, and time a small reasoning model's chain-of-thought against length-matched filler. The trace turns out to be trusted narration, not working memory: the model doesn't need its own notes, but it believes them over its own arithmetic."
aliases:
  - "reasoning-trace-authority"
  - "trace-carried-reasoning"
---

[[blog/2026-07-19-your-agent-read-the-file-but-did-it-understand-it|Your LLM Read the File. Did It Understand It?]] ended on a test: ask questions that need the LLM to connect information rather than copy it, and judge the answers. **The test of understanding is behavioural.** A reasoning model's visible thinking trace looks like a window onto that connecting step. This post tests whether it is.

**When a reasoning model writes out its steps before answering, it looks like it is showing its work.** Ask a model a question, turn on its thinking mode, and it produces a block of text: restating facts, trying an approach, correcting itself, arriving at a number, then answering. The natural reading is that this text is where the answer gets computed, that the model writes an intermediate result and reads it back later, the way a person works a problem on scratch paper.

I ran an experiment that tests whether that reading is literally true, for a small reasoning model answering questions that require combining two facts from its context. The answer came back no. But it came back no in a way that is stranger and more useful than a plain no, and the rest of this post walks the design so you can predict the result before you see it, the same path I took to find it.

## The setup: a task that needs one step of synthesis

[Qwen3-8B](https://arxiv.org/abs/2505.09388) is an 8-billion-parameter model that can run with its thinking mode on or off. I gave it 16 question pairs based on short passages. Each pair has a plain-recall question and a combination question that needs two facts from the passage put together in one step.

Turn thinking off and the model gets **5 of 16 combination questions** right. Turn thinking on and it gets **15 of 16 combination questions** right, plus all 16 of 16 plain-recall questions. Thinking clearly helps with the connecting step. That does not yet tell us whether the written derivation is what helps.

That raises the actual question this post is about. Not "does thinking help" (it obviously does), but: how does it help? What is the trace actually doing?

## Two stories, both predicting the same headline result

There are two different mechanisms that would both produce "thinking on beats thinking off", and they make opposite predictions about what happens if you interfere with the trace text itself.

- **The trace is working memory.** The model computes an intermediate value, writes it down as tokens, and reads those tokens back to produce the final answer. The written content matters causally: remove it, and the information is gone.
- **The trace is narration.** Generating tokens buys the model more sequential compute, more forward passes chained together, regardless of what those tokens say. The text is a report of work happening elsewhere; the mechanism is the number of steps, not the content of the steps.

<img class="theme-dark-only figure-center" src="/blog/assets/trace-v-stories-dark.svg" alt="Two competing stories. Working memory: the model writes an intermediate result and reads the note back to answer. Narration: extra processing steps lead to the answer and emit a written report; reading the report is not necessary." />
<img class="theme-light-only figure-center" src="/blog/assets/trace-v-stories-light.svg" alt="Two competing stories. Working memory: the model writes an intermediate result and reads the note back to answer. Narration: extra processing steps lead to the answer and emit a written report; reading the report is not necessary." />

The second story has a basis in how transformers compute. Each generated token adds a forward pass that can attend to earlier positions. [Merrill and Sabharwal's analysis of chain-of-thought expressivity](https://arxiv.org/abs/2310.07923) shows that intermediate generation can extend a fixed-depth transformer's computational power. That supports the idea that extra steps can help. It does not establish that meaningless filler is equivalent to a written derivation; that is an empirical question here.

Both stories predict exactly what I saw: thinking on beats thinking off. Neither is distinguishable from the headline number alone.

**Stop and think:** if you had to design one intervention on the trace that forces these two stories apart, what would you cut, and what would you compare it against?

## The reader's first trap: deleting text changes two things at once

The obvious move is to delete part of the trace and see if the answer breaks. Here is the trap: deleting trace text changes two things at once. It removes content, and it shortens how many generation steps happen before the answer. If accuracy drops after a cut, you cannot tell which of the two you just removed.

The fix follows [Lanham et al.'s faithfulness methodology](https://arxiv.org/abs/2307.13702): compare an early answer with a length-matched filler companion. Delete a sentence, and also run a version where that sentence is replaced by neutral filler of the same token length. The cut removes both content and steps; the filler version removes the content while preserving the length. Comparing both with the full trace helps separate a need for the written content from a need for more steps before answering.

<img class="theme-dark-only figure-center" src="/blog/assets/trace-v-filler-dark.svg" alt="A full trace contains an earlier span, a derivation and a later span. Cutting the derivation removes content and shortens the trace. Replacing it with matched-length filler removes the content while keeping the token count and later positions fixed." />
<img class="theme-light-only figure-center" src="/blog/assets/trace-v-filler-light.svg" alt="A full trace contains an earlier span, a derivation and a later span. Cutting the derivation removes content and shortens the trace. Replacing it with matched-length filler removes the content while keeping the token count and later positions fixed." />

## The interventions, and the numbers, locked in before I looked

All the numbers below are read over 12 question pairs, out of 16 total, that passed a pre-registered filter: each pair had to already answer correctly with its full reasoning trace and fail with no trace at all, so these are pairs where having a trace helped. This is a small sample.

I took one sampled reasoning trace per question, frozen, and ran every intervention against that same frozen trace, so a difference between interventions reflects the intervention and not re-sampling noise. Before running anything, I wrote down the bars a result would need to clear: a working-memory story needed accuracy to collapse to 0.3 or below once the derivation is masked, and 0.7 or above would count as evidence against it. Locking the bars first matters: a number like 0.83 can be told two opposite stories depending on which way you were already leaning, and fixing the threshold before the data exists removes that choice from hindsight.

**Masking every restatement of the derivation.** Not just one sentence: the model tends to re-derive its own intermediate result more than once across a trace ("wait, let me check that again"), so a single cut gets absorbed by its own redundant siblings. I masked every sentence that stated the derived value, using the causal attention-suppression technique from [Thought Anchors](https://arxiv.org/abs/2506.19143) (block every downstream token's attention to the target sentences, across all layers and heads). Correctness on the combination questions: **0.83** (bootstrap CI [0.58, 1.00], wide given how few questions this covers). A KL-divergence readout on the downstream tokens (averaging 5.2) confirms the mask is not inert: it measurably perturbs generation. It does not stop the model reaching the right answer.

<img class="figure-center" src="/blog/assets/thought-anchors-methods-overview.png" alt="Thought Anchors method diagram: sentences in a reasoning trace are labelled and tested with interventions. Panel B includes attention suppression, blocking later tokens from reading a selected sentence." />

*How a reasoning trace becomes an experimental surface: label the sentences, then intervene on them. The bottom box of panel B is the exact move my masking runs use: block every downstream token's attention to a target sentence and read the effect on what follows. Figure 1 of [Thought Anchors](https://arxiv.org/abs/2506.19143) (CC BY 4.0).*

**A truncation ladder, with a filler-matched companion.** I truncated the trace at increasing points and forced an answer from each truncation point, tracking where the derivation sentence sits on that ladder. On three quarters of the used questions, the answer is already correct before the derivation sentence is even reached. As the control, I ran a second ladder where the truncated remainder is replaced by content-free filler of matched length instead of being cut outright. That filler ladder crosses to correct early too, on 0.83 of questions, so the written derivation has not yet appeared when most answers become correct, even with length held fixed.

<img class="figure-center" src="/blog/assets/e7-truncation-ladder.png" alt="Truncation and filler-matched ladders plot the fraction of correct answers against the fraction of the trace retained. Most answers become correct before the first derivation sentence; top ticks mark those sentences, with a median at 62 percent of the trace." />

*Both ladders climb together, and before the derivation. Fraction of items answering correctly against how much of the frozen trace is kept; the ticks along the top mark where each item's first derivation sentence appears (median: 62 percent of the trace). The filler companion also reaches correct answers before the derivation on most items. My run data, 12 used pairs.*

**A random-sentence control.** As a check on the masking mechanic itself, rather than blocking derivation sentences, I blocked the same number of sentences chosen at random from elsewhere in the trace. This is the guard against a false positive where masking anything at all collapses the answer, which would make the derivation look causally important for no interesting reason. Correctness under the random mask: **1.00**. The masking mechanic itself is not what breaks the answer.

**Corrupting the derivation.** Rather than remove the derivation, I replaced every restatement of it with the same, single, wrong value, forcing the rest of the trace to work from a false premise. The model follows the corrupted value into its final answer on **0.92** of questions (95 percent CI roughly 0.75 to 1.00). A matched negative control, corrupting an unrelated placeholder sentence instead of the derivation, is followed 0.00 of the time and ignored 1.00 of the time, so the mechanic itself behaves cleanly.

## The reader's second trap: what does 0.83 actually mean

Before reading on: does 0.83 mean the written derivation matters a little, or that it barely matters at all? Read quickly, "masking dropped correctness to 83 percent" can sound like support for the working-memory story. The trap is to treat any drop as evidence that the model needs the written derivation.

The used questions were filtered to be correct by construction when the trace is intact and incorrect when there is no trace at all, so the baseline for these items, before any masking, is 1.00, not some unknown starting point. Masking did lower correctness from 1.00 to 0.83. But the answer survived losing access to every derivation restatement on 83 percent of items that failed without a trace in the first place. The claim under test is that those restatements are necessary, not that masking has no effect.

That is exactly why the bars were fixed before the run: 0.83 sits well above the 0.7 line that was pre-registered as evidence against the working-memory story, and nobody had to decide, after seeing the number, which way it cut.

## The verdict: refuted, with a twist

The strong working-memory story fails the locked test for this task and this model: the written derivation is not necessary on most of these questions. There is also evidence of redundant access to the facts. Two results point the same way: removing the fact from the context and removing its restatement in the trace each, on their own, leave correctness intact across all 12 used question pairs. Either copy can suffice in these tests. That supports redundancy, without identifying the exact internal path used to compute the answer.

The interesting part is what happens when the derivation is wrong instead of masked. The written derivation is a trusted input, even though it is usually unnecessary here. The model reaches the right answer without reading it on the large majority of these items. But when the notes state a specific wrong value, the model follows that value into its answer on 92 percent of items, despite being able to answer correctly with the original trace. The note is usually unnecessary, but trusted when it is present.

<img class="theme-dark-only figure-center" src="/blog/assets/trace-v-verdict-dark.svg" alt="The written derivation is usually not needed, but trusted. Over 12 filtered pairs, masking every restatement leaves 0.83 correct, versus 1.00 for a random-sentence mask. A corrupted derivation is followed on 0.92, versus 0.00 for an unrelated-placeholder control. All bars share a zero-to-one scale." />
<img class="theme-light-only figure-center" src="/blog/assets/trace-v-verdict-light.svg" alt="The written derivation is usually not needed, but trusted. Over 12 filtered pairs, masking every restatement leaves 0.83 correct, versus 1.00 for a random-sentence mask. A corrupted derivation is followed on 0.92, versus 0.00 for an unrelated-placeholder control. All bars share a zero-to-one scale." />

## Scope: a map of a boundary, not a general law

This holds for one 8-billion-parameter model, on questions requiring exactly one step of synthesis. The truncation ladder crosses to a correct answer at or after the derivation on 0.25 of the used questions. That leaves room for the written derivation to matter on some items; the masking result says it is unnecessary on most of these items, at this depth. A three-seed check on a handful of items found the masking result stable across different sampled traces (moving by at most one item out of three across reseeds), which gives a limited check on sensitivity to the sampled trace.

[Lanham et al.](https://arxiv.org/abs/2307.13702) found trace-dependence varies by task and shrinks further with model scale on some tasks. The theory shows what extra intermediate steps can make possible; it does not settle whether the written content matters causally on a particular task. On a problem that needs more composition steps than tested here, the trace may have to carry intermediate results. Whether that is true is empirical per task, not settled by this result. This is a map of where one boundary sits, not a law about reasoning traces in general.

## Three doors this opens

**Faithfulness reads differently if the trace has authority without necessity.** If the trace narrates a computation happening elsewhere but the model still trusts whatever the trace says, then reading a trace tells you what the model will act on, not how it actually arrived at the answer. That distinction matters directly for the faithfulness literature, including recent benchmarking work like [FaithCoT-Bench](https://arxiv.org/abs/2510.04040).

**A trusted note can steer the answer.** Following a corrupted derivation on 0.92 of questions, against a clean control, shows a way to steer this model: write the value you want into the trace, and the model tends to adopt it, whether or not it needed the trace to get there in the first place.

**This bears on the case for latent, non-token reasoning.** Designs like [COCONUT](https://arxiv.org/abs/2412.06769) replace the written trace with continuous latent iterations, aiming to get the extra sequential compute without paying for it in tokens. The standard objection is that you lose the interpretable trace. That objection weakens if the trace is mostly narration to begin with. But the editable text this result highlights is exactly what a latent design gives up: there is no longer a note the model can trust or distrust, correctly or otherwise. The open question is where the serial computation has to live: in tokens you can read and edit, or in latent iterations you cannot.

The attention plot below shows why the trace is worth intervening on: some sentences draw strong attention from later positions. That locates a read; the masking and corruption tests ask what that read does to the answer.

<img class="figure-center" src="/blog/assets/thought-anchors-receiver-heads.png" alt="Thought Anchors receiver-head attention plot. A few reasoning sentences receive strong attention from later positions, visible as peaks in the plot and dark vertical stripes in the inset attention matrix." />

*A receiver head at work: a few sentences in a reasoning trace draw outsized attention from everything downstream (the spikes, and the dark vertical stripes in the inset matrix). Figure 4 of [Thought Anchors](https://arxiv.org/abs/2506.19143) (CC BY 4.0).*

## The shape of the answer

I asked whether the model reads its own thinking trace, and the answer refused to be a yes or a no. It usually does not need the written derivation: most answers survive when every restatement is masked. It does trust the derivation: write a value into it and the model adopts it against a computation it performs correctly on its own. "Showing its work" turns out to be the wrong frame entirely. The trace is closer to a briefing the model writes for itself and then believes.

Neither the plain answer surface nor the trace surface, on its own, tells you whether a model actually combined two facts or landed on the right answer by another route. That gap, between what a trace shows and what a model does, is where measurement work has to live.

---

*Revised on 2026-09-28: new figures, a corrected thinking-on comparison (15 of 16 combination questions), and a new opening that follows the revised previous post.*

**References.**

- Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning," [arXiv 2307.13702](https://arxiv.org/abs/2307.13702).
- Bogdan et al., "Thought Anchors: Which LLM Reasoning Steps Matter?", [arXiv 2506.19143](https://arxiv.org/abs/2506.19143).
- Merrill, Sabharwal, "The Expressive Power of Transformers with Chain of Thought," [arXiv 2310.07923](https://arxiv.org/abs/2310.07923).
- Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space" (COCONUT), [arXiv 2412.06769](https://arxiv.org/abs/2412.06769).
- Shen et al., "FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning," [arXiv 2510.04040](https://arxiv.org/abs/2510.04040).
- Qwen Team, "Qwen3 Technical Report," [arXiv 2505.09388](https://arxiv.org/abs/2505.09388).
