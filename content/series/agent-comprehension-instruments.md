---
title: "Agent Comprehension Instruments"
description: "Building, from scratch, the instruments that measure whether an agent actually uses and understands its context, and what each one's failure modes are."
---

An agent that reads its context and produces an answer leaves two separate questions open: did the answer lean on the right source, and did the agent get that source right. Each post in this series builds one instrument from scratch, follows the exact reasoning path to get there, and spends real space on what the instrument cannot tell you.

## Posts in this series

1. **[[blog/2026-07-12-did-your-agent-actually-read-that-file|Did Your Agent Actually Read That File?]]** - Ablation attribution: grade a frozen answer with a source in and out of context to measure which sources it leaned on, for the cost of ~64 forward passes and a linear regression.
2. **[[blog/2026-07-19-your-agent-read-the-file-but-did-it-understand-it|Your Agent Read the File. Did It Understand It?]]** - Attribution finds the right source at rank one and the answer can still be wrong. Scoring what an attention head writes, not just where it looks, is how you catch a misread.
3. **[[blog/2026-07-22-does-a-reasoning-model-actually-read-its-own-thinking-trace|Does a Reasoning Model Actually Read Its Own Thinking Trace?]]** - Cut, corrupt, and time the chain-of-thought against length-matched filler. The trace turns out to be trusted narration, not working memory, and both halves of that finding matter.
4. **[[blog/2026-07-24-is-half-your-context-window-just-marketing|Is Half Your Context Window Just Marketing?]]** - A context window is a claim about attention capacity, not recall. Needles built so nothing can be grepped put the real operating range at roughly half the number on the box, and the lexical-overlap twin shows exactly what dies at depth: the hop, not the fetch.
