---
title: "LLM Comprehension"
description: "Measuring what a language model actually does with the text you give it, one method per post."
aliases:
  - "series/agent-comprehension-instruments"
---

When a language model reads a context and answers, two questions stay open: did the answer lean on the right source, and did the model get that source right? Each post in this series builds one way to measure this from scratch, shows the reasoning behind it, and spends real space on what the measurement cannot tell you.

## Posts in this series

1. **[[blog/2026-07-12-did-your-agent-actually-read-that-file|Did Your LLM Actually Read That File?]]** - Ablation attribution (ContextCite): keep the answer fixed, remove sources, and fit a regression to see which ones the answer rested on. On planted-fact tests it ranks the right source first in 55 of 55 runs.
2. **[[blog/2026-07-17-how-attention-works|How Does an LLM Read Its Context?]]** - One fact followed through tokens, attention and the residual stream. Where a head reads and what it writes are separate, and the head that looks hardest at a fact is not always the one that pushes the answer.
3. **[[blog/2026-07-18-which-heads-read-your-context|Which Heads Read Your Context?]]** - Three published detectors for the heads that carry a fact into the answer, worked by hand on one example: a copy test, question-driven attention, and the push toward the answer each pick a different head, and a real model splits the same way.
4. **[[blog/2026-07-19-your-agent-read-the-file-but-did-it-understand-it|Your LLM Read the File. Did It Understand It?]]** - The write-score heads are the ones an inferred answer needs (switch them off and the score drops to zero), and attention knockout confirms the answer reads the fact. But as a meter for answer quality the write score failed: clean answers scored lower than flawed ones.
5. **[[blog/2026-07-22-does-a-reasoning-model-actually-read-its-own-thinking-trace|Does a Reasoning Model Actually Read Its Own Thinking Trace?]]** - Cut, corrupt, and time the chain-of-thought against length-matched filler. The trace turns out to be trusted narration, not working memory, and both halves of that finding matter.
6. **[[blog/2026-07-24-is-half-your-context-window-just-marketing|Is Half Your Context Window Just Marketing?]]** - Qwen3-8B advertises 32K tokens. On questions that share no words with the fact they need, it is perfect to 8K and usable to about 16K, half the window, while a keyword twin of the same fact is still found at 24K.
7. **[[blog/2026-08-12-does-precision-loss-break-retrieval-or-does-distance|Does Precision Loss Break Retrieval, or Does Distance?]]** - The same two-fact, no-shared-words task run down a quantization ladder and out a length ladder. Precision is innocent to roughly 3.5 bits per weight; distance is already breaking retrieval inside the native window. And the one real precision failure is confident and wrong, not uncertain.
8. **[[blog/2026-09-27-is-half-your-context-window-still-marketing|Is Half Your Context Window Still Marketing in 2026?]]** - The same no-shared-words test on a 2026 frontier model, GPT-6 Luna, from 250 tokens to 512K. Its usable range is about 64K (128K with reasoning on) of an advertised million, well before Codex's default compaction point, and what that means for where you set yours.
