---
title: "Is Half Your Context Window Still Marketing in 2026?"
date: 2026-09-27
draft: true
series: "LLM Comprehension"
series_order: 6
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

<img class="theme-dark-only figure-narrow" src="/blog/assets/luna-context-curve-dark.svg" alt="Accuracy of GPT-6 Luna against context length: the shared-word control stays near 100%; recall by meaning with reasoning off falls from about 90% to 46% at 224K; with reasoning high it holds near 85% to 128K, then falls to 26% at 512K. Codex's 272K default window and the advertised 1M are marked." />
<img class="theme-light-only figure-narrow" src="/blog/assets/luna-context-curve-light.svg" alt="Accuracy of GPT-6 Luna against context length (light version of the same chart)." />

*How often Luna finds a planted fact as the context grows. The top line is a control question that
shares words with the fact; the two lower lines need the fact to be connected by meaning. Each point
is 28 to 168 questions; the bands show the uncertainty from that sample size.*

## Two ways to find a fact

Plant one sentence in a long document and ask about it in two ways:

<img class="theme-dark-only figure-narrow" src="/blog/assets/luna-v-two-paths-dark.svg" alt="The planted fact 'Kai lives next to the Rijksmuseum.' The word-match question 'Who lives next to the Rijksmuseum?' goes straight from the shared word to Kai. The meaning question 'Who has been to Amsterdam?' shares no words and needs a hop: Amsterdam, then Rijksmuseum, then Kai." />
<img class="theme-light-only figure-narrow" src="/blog/assets/luna-v-two-paths-light.svg" alt="The same two lookups in the light theme." />

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
model, Qwen3-8B, kept only about half of its 32K window. The question now: has frontier training
fixed this?

## The experiment

I ran NoLiMa's published questions on Luna, in documents from 250 tokens to 512K:

<img class="theme-dark-only figure-narrow" src="/blog/assets/luna-v-experiment-dark.svg" alt="The experiment in three steps: grow the document from 250 tokens to 512K; place the fact at 25%, 50% or 75% depth; ask both a meaning question and a shared-words question. With the fact removed, Luna named the right character 0 times out of 28." />
<img class="theme-light-only figure-narrow" src="/blog/assets/luna-v-experiment-light.svg" alt="The experiment diagram in the light theme." />

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

## So where should compaction go?

<img class="theme-dark-only figure-narrow" src="/blog/assets/luna-v-compaction-dark.svg" alt="Compaction guide: the useful ceiling for meaning-based work is 64K with reasoning off and about 128K with reasoning high. On the full 1M bar, everything past 512K is unmeasured. Zoomed on Codex's 272K default window, the default compaction point at about 258K sits well inside the degrading zone." />
<img class="theme-light-only figure-narrow" src="/blog/assets/luna-v-compaction-light.svg" alt="The compaction guide in the light theme." />

Codex's default compacts Luna at about 258K, deep in the zone where recall by meaning has already
lost a third to a half of its accuracy. My half-full rule lands close to the 128K line. What I do
now:

- **Compact at about 128K** for work that connects ideas across the session, or 64K with
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

**References.**

- Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze, "NoLiMa: Long-Context
  Evaluation Beyond Literal Matching," [arXiv 2502.05167](https://arxiv.org/abs/2502.05167);
  [dataset](https://huggingface.co/datasets/amodaresi/NoLiMa).
- Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?"
  [arXiv 2404.06654](https://arxiv.org/abs/2404.06654).
- [OpenAI Codex CLI](https://github.com/openai/codex) (the input limit is `MAX_USER_INPUT_TEXT_CHARS`
  in `codex-rs/protocol/src/user_input.rs`).
