---
title: "Agent Comprehension Instruments"
description: "Building, from scratch, the instruments that measure whether an agent actually uses and understands its context, and what each one's failure modes are."
---

An agent that reads its context and produces an answer leaves two separate questions open: did the answer lean on the right source, and did the agent get that source right. Each post in this series builds one instrument from scratch, follows the exact reasoning path to get there, and spends real space on what the instrument cannot tell you.

## Posts in this series

1. **[[blog/2026-07-12-did-your-agent-actually-read-that-file|Did Your Agent Actually Read That File?]]** - Ablation attribution: grade a frozen answer with a source in and out of context to measure which sources it leaned on, for the cost of ~64 forward passes and a linear regression.
2. **[[blog/2026-07-19-your-agent-read-the-file-but-did-it-understand-it|Your Agent Read the File. Did It Understand It?]]** - Attribution finds the right source at rank one and the answer can still be wrong. Scoring what an attention head writes, not just where it looks, is how you catch a misread.
