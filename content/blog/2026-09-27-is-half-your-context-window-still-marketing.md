---
title: "Is Half Your Context Window Still Marketing in 2026?"
date: 2026-09-27
draft: false
series: "LLM Comprehension"
series_order: 8
tags:
  - llm
  - agents
  - long-context
description: "GPT-6 Luna advertises a 1M-token context. On questions that need the model to connect a fact by meaning rather than by matching words, it holds up to about 64K tokens, about 128K with reasoning on, and is below half its short-context accuracy by 512K. What that means for where you set compaction."
---

For a long time I have had a rule of thumb for coding agents: once the context is about half full,
the model gets noticeably dumber. It still answers and still sounds confident, but it starts missing
things it would have caught an hour earlier: a constraint from the start of the session, a file it
read twenty tool calls ago. So I set my agents to compact at around half the window, and I try to
wrap up a task before they get there.

That rule was a hunch. Here it is as numbers for GPT-6 Luna, a current frontier model advertised
with a one-million-token window. When the model has to connect ideas rather than match words, its
useful range is about **64K tokens**, or about **128K with reasoning on**. The hunch holds, and for
this kind of recall it is, if anything, too generous.

## Two ways to find a fact

In an invented example, plant one sentence in a long document and ask about it in two ways:

<img class="theme-dark-only figure-center" src="/blog/assets/luna-v-two-paths-dark.svg" alt="An invented example: Kai lives next to the Rijksmuseum. The question Who lives next to the Rijksmuseum goes from the shared word Rijksmuseum to Kai. Who has been to Amsterdam needs the connection from Amsterdam to Rijksmuseum before retrieving Kai." />
<img class="theme-light-only figure-center" src="/blog/assets/luna-v-two-paths-light.svg" alt="An invented example: Kai lives next to the Rijksmuseum. The question Who lives next to the Rijksmuseum goes from the shared word Rijksmuseum to Kai. Who has been to Amsterdam needs the connection from Amsterdam to Rijksmuseum before retrieving Kai." />

The first question works like a search box. The second needs one small step of reasoning while
reading: the Rijksmuseum is in Amsterdam, so the sentence about Kai is the answer.

That second kind is what you need from an LLM most of the time. You rarely ask about the exact words in a
file; you ask what the file means for your problem. The config says the service sits behind a load
balancer, and your question is why outside requests time out.

Most "needle in a haystack" tests only ask the first kind, which is why models ace them at any
length. The NoLiMa benchmark (Modarressi et al., [arXiv 2502.05167](https://arxiv.org/abs/2502.05167))
asks the second kind, with invented names so the model cannot know the answer from training. In
2025 it found GPT-4o, near perfect on short documents, at about 70% by 32K tokens. In
[[blog/2026-07-24-is-half-your-context-window-just-marketing|the previous post on this]] a small open
model, Qwen3-8B, kept only about half of its 32K window on my item set. The question now: has frontier training
fixed this?

## The experiment

I ran NoLiMa's published questions on Luna, in documents from 250 tokens to 512K:

<img class="theme-dark-only figure-center" src="/blog/assets/luna-v-experiment-dark.svg" alt="Document lengths grow from 250 tokens to 512K, shown schematically. Separate runs put one fact at 25%, 50% or 75% depth and ask a meaning question or a shared-word control. With the fact removed, Luna names the right character 0 times out of 28." />
<img class="theme-light-only figure-center" src="/blog/assets/luna-v-experiment-light.svg" alt="Document lengths grow from 250 tokens to 512K, shown schematically. Separate runs put one fact at 25%, 50% or 75% depth and ask a meaning question or a shared-word control. With the fact removed, Luna names the right character 0 times out of 28." />

What the figure does not show:

- **A clean setup.** Luna ran through the Codex command-line tool with web search, shell and
  project files off, and NoLiMa's one-line system message in place of Codex's. No answer used a
  tool.
- **One workaround past 224K.** Codex refuses a single message over about one million characters
  (roughly 247K tokens), a limit of the tool, not the model. For 384K and 512K the document went
  into the system instructions instead; at 192K, where both fit, the two placements scored within 4
  points of each other.
- **The cost.** 2,560 questions, about 32 dollars at Luna's public API prices, most of it the
  384K and 512K prompts. On a ChatGPT subscription through Codex it used at most 6% of a week's
  allowance.
- **The scope.** One model, two source books, one answer per question. Read the result as a shape,
  not decimals to quote.

## What happened

**The control never falls.** When the question shares words with the fact, Luna finds it at 93% or
better at every length. The model can still *see* everything in its window.

**Recall by meaning falls long before the window ends.** NoLiMa calls a length usable while the
model keeps 85% of its short-context score. With reasoning off Luna starts at 92%, so the bar is
0.85 × 92 ≈ 78%. It clears that at 64K (86%) and misses at 128K (67%), and from there it only
falls. (A dip to 77% at 16K recovers by 64K, so I read it as noise.) Usable range: **64K**, about 6%
of the advertised million.

**Reasoning buys time, not a fix.** With reasoning high the bar is 0.85 × 93 ≈ 79%. Luna clears it
at 128K (82%) and misses at 192K (76%). Past that the same fall returns, down to 26% at 512K.

<img class="theme-dark-only figure-center" src="/blog/assets/luna-v-curve-dark.svg" alt="Luna accuracy versus context length on a log scale. Shared words stay at 93% or better through 224K. Meaning with reasoning off scores 86% at 64K, 67% at 128K and 46% at 224K. Meaning with reasoning high scores 82% at 128K and 26% at 512K. Bands show 95% Wilson intervals. The 272K Codex window and advertised 1M are marked; no observations extend to 1M." />
<img class="theme-light-only figure-center" src="/blog/assets/luna-v-curve-light.svg" alt="Luna accuracy versus context length on a log scale. Shared words stay at 93% or better through 224K. Meaning with reasoning off scores 86% at 64K, 67% at 128K and 46% at 224K. Meaning with reasoning high scores 82% at 128K and 26% at 512K. Bands show 95% Wilson intervals. The 272K Codex window and advertised 1M are marked; no observations extend to 1M." />

*How often Luna finds a planted fact as the context grows. The top line is a control question that
shares words with the fact; the two lower lines need the fact to be connected by meaning. Each point
is 28 to 168 questions; the bands show the uncertainty from that sample size.*

**The failures are silent.** At long context Luna does not name the wrong person; it names no one.
That is the worst kind of miss: the fact was in the context, and the model carried on
as if it was not.

> [!note]- All the numbers
> | Context | Meaning, reasoning off | Meaning, reasoning high | Shared words (control) |
> |---|---|---|---|
> | up to 1K | 92% | 93% | 97% |
> | 16K | 77% | 85% | 100% |
> | 64K | 86% | 90% | 96% |
> | 128K | 67% | 82% | 96% |
> | 192K | 57% | 76% | 93% |
> | 224K | 46% | 63% | 96% |
> | 384K | | 57% | |
> | 512K | | 26% | |
>
> The reasoning-high rows at 16K, 64K and 224K use one source book, the others two.

That is real progress since 2025, when GPT-4o's usable range on this test was well under 32K. But
holding a million tokens and reasoning over them are still two different things.

## It is not just Luna

Luna is the only model I measured here, but nothing in the result is specific to it. The NoLiMa
authors tested 13 models that claim at least 128K tokens of context: at 32K, 11 of them fell below
half of their short-context score, and neither reasoning models nor chain-of-thought prompting held
up. RULER (Hsieh et al., [arXiv 2404.06654](https://arxiv.org/abs/2404.06654)) found the same shape
on 17 models and a wider set of tasks: nearly perfect on the plain needle test, yet only half of
them still performed well at 32K. The earlier post found it on Qwen3-8B.

NoLiMa's authors trace the cause to attention itself. Without a shared word to latch onto, picking
out the one relevant sentence gets harder the more text surrounds it. As far as I know, no model or
technique has removed this. Newer models push the drop further out; none has made it go away.

So the question for any model is not *whether* it falls, but *how early*. I have not measured
Claude Opus 5.5 or the other GPT-6 models (Sol, Astra) this way. In daily use their drop comes early
too, around half the window. That is an impression, not a measurement, but it matches every
measurement above.

## So where should compaction go?

Codex's default compacts Luna at about 258K, deep in the zone where recall by meaning has already
lost a third to a half of its accuracy. My half-full rule lands close to the 128K line: half of
Codex's 272K window is 136K.

<img class="theme-dark-only figure-center" src="/blog/assets/luna-v-compaction-dark.svg" alt="Two linear scales: the advertised 1M capacity is measured only through 512K, with the rest hatched. A zoom into the 272K Codex window shows usable context of 64K with reasoning off and about 128K with reasoning high. Default compaction at about 258K falls past both. Usable means retaining at least 85% of the short-context score." />
<img class="theme-light-only figure-center" src="/blog/assets/luna-v-compaction-light.svg" alt="Two linear scales: the advertised 1M capacity is measured only through 512K, with the rest hatched. A zoom into the 272K Codex window shows usable context of 64K with reasoning off and about 128K with reasoning high. Default compaction at about 258K falls past both. Usable means retaining at least 85% of the short-context score." />

What I do now:

- **Compact at half the window, whatever the model.** That is where the Luna numbers put it with
  reasoning on, and a reasonable default for models nobody has measured this way.
- **For Luna, compact at about 128K** for work that connects ideas across the session, or 64K with
  reasoning off. Compact by hand when a task changes direction.
- **Keep reasoning on in long sessions.** It doubles the usable range.
- **Restate what matters, in the same words.** Word matching survived every length, so repeating a
  key constraint where it is needed turns a meaning lookup back into a word match.

A million tokens is how much text the model can take in. Set your compaction by how much it can
still reason over.

## Notes on how this was made

I designed, ran and reviewed these experiments with the assistance of AI agents (Claude and
Codex); every number in this post was checked against the stored answers by an independent
review before I wrote it down. The test harness builds on a small research library that is not
public yet.

*Revised on 2026-09-27: new figures.*

**References.**

- Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze, "NoLiMa: Long-Context
  Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167);
  [dataset](https://huggingface.co/datasets/amodaresi/NoLiMa).
- Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?"
  [arXiv 2404.06654](https://arxiv.org/abs/2404.06654).
- [OpenAI Codex CLI](https://github.com/openai/codex) (the input limit is `MAX_USER_INPUT_TEXT_CHARS`
  in `codex-rs/protocol/src/user_input.rs`).
