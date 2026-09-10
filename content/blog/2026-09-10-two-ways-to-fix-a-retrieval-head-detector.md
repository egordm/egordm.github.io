---
title: "Two Ways to Fix a Retrieval-Head Detector"
date: 2026-09-10
draft: true
tags:
  - llm
  - agents
  - interpretability
  - long-context
description: "Wu's copy-match test for retrieval heads breaks on any question that cannot be answered by copying. Part 2 of this series took one exit: score what a head writes instead of where it looks. There is a second exit on the other axis, taken by a different lab that never cites the first, and laying the two side by side turns three published methods into one instrument with two knobs."
aliases:
  - "detector-frames"
  - "qrhead-vs-locos"
series: "Agent Comprehension Instruments"
series_order: 6
---

[[blog/2026-07-19-your-agent-read-the-file-but-did-it-understand-it|Part 2 of this series]] built a detector for the heads that carry non-literal retrieval, and it opened with a broken instrument. Wu et al.'s retrieval-head test asks whether the token a head attends to hardest is the token the model then emits. On needle-in-a-haystack that works. On a question whose answer has to be synthesized rather than copied, it collapses: the top-head score falls from 0.97 to 0.03, and ablating the heads it selects can make the task *better*, which is the signature of selecting causally irrelevant heads.

That post's fix was to change what you measure: score the OV write onto the answer direction, not the QK read. It works, and the ablation numbers are decisive.

What that post did not say, because the paper it follows does not say it either, is that **this is one of two available fixes, on two different axes.** A different lab took the other one a year earlier. The two branches descend from the same parent, and as far as their reference lists go, neither knows the other exists. Putting them in one frame is what this post does, and the frame has a hole in it.

## The instrument, before any of the three fill it in

Strip Wu, QRHead and LOCOS down and the same four steps appear in all of them:

1. compute a quantity per (source token, head)
2. sum it over a labeled span
3. subtract a baseline
4. emit one score per head, rank all heads, keep the leading one to two percent

Every published difference between the three lives in step 1 or step 3. Nothing else moves. Once you see that, "which heads are the retrieval heads" stops being a question about the model and becomes a question about which two slots you filled in.

## Step 1, the operand: what a head gets credit for

**Wu's operand is a coincidence test on token identity.** At the decode step that emits token $y_t$, take head $h$'s attention row, take its argmax over context positions, look up which token sits there, and check whether that token is $y_t$, copied from the context. Credit is 1 or 0. The head's score is how often that coincidence happens across a dataset.

Two properties of that operand matter more than they first look:

**It is winner-take-all.** A head that spreads half its mass across five copies of the right answer scores zero if its argmax lands on the word "the". Wu reads the mode, not the mass.

**It is conditioned on the model being right.** $y_t$ is what the model *emitted*. If the model answers wrongly, a head that attended perfectly to the correct source scores zero, for reasons that have nothing to do with attention. This is the property that decides where the instrument can be pointed, and we will come back to it.

It is worth being precise about how Wu fails on a synthesis question, because the obvious guess is wrong. The guess is that the answer is not in the context, so the test is undefined. Usually it *is* in the context: in a NoLiMa-style item, the answer is an entity sitting in the haystack, and the model copies it out once it has worked out which one. Wu's test is perfectly well defined. It just goes quiet, because the model is usually wrong, and because the argmax is a brittle read of a distributed pattern. Defined and uninformative is a different failure from undefined, and it is the more dangerous one, because it still returns a number.

**LOCOS's operand keeps the attention and multiplies it by a write.** Per source position $j$ at decode step $t$:

$$\phi^{(l,h)}_{t,j} = \alpha^{(l,h)}_{t,j} \cdot u_{y_t}^\top W_O^{(l,h)} v^{(l,h)}_{t,j}$$

Part 2 builds this bottom-up. The compression worth carrying is that $\phi$ is **read relevance times write relevance**: how much the head looked at this position, weighted by how useful what it read turned out to be. Note that $\alpha$ is still in there. LOCOS does not replace attention, it re-weights it.

**QRHead's operand is the attention mass, and nothing else.** Drop the copy-match requirement entirely. Score a head by how much attention it sends from the query tokens to a source you have labeled relevant, summed over that source's tokens.

That substitution is the whole of QRHead's method. It is a small change with a large consequence: Wu's criterion needs a literal string to copy, which is why it is structurally a needle-in-a-haystack instrument. QRHead's criterion needs only a relevance label, which is why it runs on real multi-hop QA and on BEIR, where nothing is copied verbatim.

## Two different things called top-K

The natural reading of QRHead, once you hear "mass instead of a binary", is that it relaxes Wu's argmax into a top-K over positions. It does not, and the difference is structural rather than cosmetic.

There are two top-K operations in this literature and they sit at different levels:

- **Over heads.** Both methods rank every head and keep the leading one to two percent. QRHead uses 16 heads for models below 10B parameters and 32 for Llama-3.1-70B. This is shared, not a difference.
- **Over positions.** Wu takes the argmax, so top-1. QRHead takes **no top-K over positions at all.** It does not rank positions. It sums mass over a span it was *told* about.

That last point is the real dividing line, and it is a swap of supervision rather than a relaxation:

- **Wu discovers** which position mattered, then checks it against the emitted token. The model's own output is the verifier.
- **QRHead is told** which span is relevant, then measures how much mass landed there. The relevance label is the verifier.

Wu needs no annotation but requires a literal copy. QRHead needs no copy but requires labels. Neither is free.

## Is the label a real dependency, or just a way to save memory?

A fair objection at this point: marking the ground-truth span sounds like a reduction applied for convenience. You could compute attention to every position, keep the whole thing, and look at where the head goes without telling it the answer first. Is the label load-bearing, or is it an accountant's choice?

For the measurement, **it is an accountant's choice**, and the numbers say so more strongly than the intuition does. The naive move of asking a transformer for all its attention weights is genuinely impossible at these lengths: on an 8B model at 16k tokens, a single layer's full attention matrix is about 16 GB, and all its layers together run to hundreds. But the expensive axis there is context-against-context, and QRHead's released implementation already kills it by recomputing attention only for the *query* rows against the full key cache. What it produces is shaped

```
(layers, heads, query_tokens, context_tokens)
```

which is about 1.8 GB across all layers at 16k, and which is already "where does every head look, across the whole context", with no label anywhere in it. The span sum is applied afterward. You could keep that tensor and browse it.

For **detection**, the label is not optional and no amount of memory buys you out. Scoring a head means summing the *relevant* sources and averaging over labeled examples. With the unlabeled tensor you can see where every head looks, but you have no criterion for which head is good. "This head attends hard to position 4,912" is not a ranking until something says whether position 4,912 deserved it.

So the honest split is that QRHead's label is a reduction at measurement time and an objective at detection time. Which also clarifies what the method produces, because it produces two different things pointing in opposite directions. **Detection** consumes labeled examples and emits per-head scores, which is a statement about the *model*, computed once and reused. **Retrieval**, which is the paper's headline application and the reason its title ends in "and Re-ranking", consumes an already-detected head set and emits per-source scores for a new query, which is a statement about *this context*. The heads are the instrument. The source ranking is the reading.

## Step 3, the baseline: two contrasts that defend against different things

This is the axis nobody crosses, and it is the more interesting half.

**QRHead subtracts a null query.** Run the same context again with the query replaced by something contentless (the paper uses "N/A"), and subtract. Same context, different question.

What that defends against is a head with a positional habit. Raw attention mass is dominated by sinks, by the beginning of the context, and by recency. Without the subtraction, a head that always parks on the first few tokens scores as a brilliant retriever whenever the answer happens to sit early. The null query isolates the part of the attention that is driven by the *content* of the question.

**LOCOS subtracts a length-rescaled off-source sum.** Sum $\phi$ over the needle positions to get $\Phi^+$, sum it over everything else, rescale that by needle length over non-needle length, call it $\Phi^-$, and score the head on $\Phi^+ - \Phi^-$. Same question, different positions.

The rescaling is what makes the comparison meaningful. The needle is maybe 20 tokens and the background is 5,000, so a raw comparison of two sums is decided by volume alone. Multiplying by $|\text{needle}| / |\text{non-needle}|$ turns $\Phi^-$ into the $\phi$ mass you would expect from an *average slice of the context the same size as the needle*. The contrast then reads: how much more does this head concentrate here than on an ordinary piece of the same context. Zero means the needle is unremarkable.

What that defends against is subtler than a positional habit. Consider a model that answers correctly from its own weights, having memorized the fact, and never really uses the planted copy at all. The write term still points at the right token, so $\phi$ over the needle looks healthy. But so does $\phi$ over any other slice, because nothing is concentrating on the needle. The contrast goes to zero. **The off-source term is what makes it a context-use measure rather than a correctness measure.**

Lay them out and the orthogonality is obvious:

| Baseline | Held fixed | Varied | Defends against |
|---|---|---|---|
| Null query | the context | the question | a head that reads here regardless of what you asked |
| Off-source | the question | the positions | a head that writes strongly everywhere, and an answer produced from memory rather than context |

Neither method controls for the other's confound. Not because either was careless, but because each branch inherited one baseline from its own parent and never met the other branch.

## The grid

Two slots, two published options each:

| | Position contrast (off-source) | Query contrast (null query) |
|---|---|---|
| **Attention only** | LOCOS's attention-only control | **QRHead** |
| **$\phi$ = read $\times$ write** | **LOCOS** | *unoccupied* |

Three cells are published. The top-left is not a method anyone shipped; it is the control LOCOS ran on itself, applying its own spatial contrast to raw attention weights instead of the logit projection, in order to prove that the OV projection rather than the contrast was doing the work.

The bottom-right cell is empty. And a design carrying *both* contrasts is not on the grid at all.

## Why the hole is there

The two branches descend from the same paper and never meet.

- **Princeton PLI and NYU:** Wu, then QRHead, then DySCO. Attention mass, query-focused, pointed at retrieval and re-ranking quality.
- **Edinburgh:** Wu, then LOCOS. Write direction, pointed at detecting a head class that literal criteria miss.

LOCOS's retrieval-head references are Wu et al., three KV-cache-compression papers (Fu et al., CompressKV, DuoAttention) and the circuits foundations. It cites no query-focused retrieval-head work at all. QRHead predates LOCOS by roughly thirteen months and cannot cite it. The branches are reading different literatures: one is talking to people who want better retrieval, the other to people who want smaller caches.

There is a nice piece of convergent evidence sitting in the gap. Both branches independently report that their heads barely overlap Wu's: under 25 percent at top-32 for QRHead, and 2 of the top 10 on Qwen3-8B for LOCOS. Two labs, two sides of the circuit, one shared conclusion that neither of them states in the other's terms. **"Retrieval head" names a criterion, not a fixed set of heads.** Change the criterion and you get a different set out of the same model.

## The empty cell is cheaper than it looks

The obvious objection to filling it is cost. If you want $\phi$ calibrated against a null query, you need the write term measured under both the real query and the null one, and that sounds like doubling an already expensive capture.

It is not, and the reason is a property of the prompt layout rather than a clever trick.

In this setup the sources sit inside a shared context and the query is a suffix. So the source tokens causally *precede* the query. Their value vectors cannot see it. Swap the real query for "N/A" and every $v_j$ is unchanged, and the unembedding row $u_y$ is a fixed parameter, so it is unchanged too. Only $\alpha$ moves. Which means:

$$\phi_j - \phi_j^{\text{null}} = \left(\alpha_j - \alpha_j^{\text{null}}\right) \cdot u_{y}^\top W_O v_j$$

**The query calibration factors straight through the write term.** One null pass calibrates the product. You do not need a second null for the write.

Better, this is not a property you have to arrange. QRHead's implementation already computes the context KV once and reuses it for both passes, so the value vectors are literally the same cached tensors rather than merely equal in principle.

And the other contrast is free outright: the off-source term is a change of index set within a forward pass you already ran, not a replay. So a detector carrying **both** baselines costs one extra forward pass in total, not two captures and a doubling.

The real cost is a label, not compute. $\phi$ needs the answer token's identity, not merely its span, because something has to supply $u_y$. QRHead alone needs only "this span is relevant". That is a strictly stronger annotation requirement, and it should be stated up front rather than discovered halfway through a run.

One thing that requirement buys back, though, is worth the trade. If you project onto the *intended* answer's unembedding row rather than the emitted token's, the score stays defined on items the model got wrong. That undoes the property that broke Wu.

## Where this breaks

This post has no numbers of its own. Every figure quoted above is published, from the QRHead and LOCOS papers and their released code, and the factoring argument in the previous section is a derivation from the prompt layout that neither paper states and that nobody has yet checked numerically. A derivation that looks clean on paper is a hypothesis about an implementation until someone runs it. Treat it as one.

Two further limits are worth naming before anyone points this at real work.

**The read-only quantity has one negative result against it on exactly this construct class.** LOCOS's attention-only control is the top-left cell of the grid, and it came back weak: the OV projection, not the spatial contrast, carried the detection on the non-literal task. That control used a position contrast rather than a query contrast, so it does not settle what a query-calibrated attention read does. But it is the nearest published measurement, and it is not encouraging for measuring attention alone on synthesis-style questions.

**A null reading has three explanations, not two.** Point a fixed head set at a long context, watch attention to the correct source fall to nothing at depth, and the tempting conclusion is that retrieval failed. Two other readings survive that observation. The span may still be supplied while a later stage fails, which no supply measurement can distinguish, because attention received is not the same as attention used. Or the read-only quantity may not discriminate on this construct class at all, per the paragraph above. And if the head set was detected on a literal task and then pointed at a non-literal one, there is a fourth: those may never have been this construct's heads, and they would read low at every depth. Separating these needs an observation the supply read does not contain, such as a shallow-depth reading where behavior is known to be intact, or a lexical control measured on the same head set.

That is the series' usual ending, and it applies with more than usual force here, because a grid with a hole in it is a very good way to convince yourself that filling the hole is the same thing as learning something.

## References

- Zhang, Yin, Yen, Chen, Ye, "Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking" (QRHead), [arXiv 2506.09944](https://arxiv.org/abs/2506.09944), EMNLP 2025; code at [github.com/princeton-pli/QRHead](https://github.com/princeton-pli/QRHead).
- Gema, Alex, Minervini, "Logit-Contribution Scoring Identifies Non-Literal Retrieval Heads" (LOCOS), [arXiv 2607.01002](https://arxiv.org/abs/2607.01002); code at [github.com/aryopg/locos](https://github.com/aryopg/locos).
- Wu, Wang, Xiao, Peng, Fu, "Retrieval Head Mechanistically Explains Long-Context Factuality," [arXiv 2404.15574](https://arxiv.org/abs/2404.15574).
- Ye, Zhang, Yin, Yen, Chen, "DySCO: Dynamic Attention-Scaling Decoding for Long-Context Language Models," [arXiv 2602.22175](https://arxiv.org/abs/2602.22175); code at [github.com/princeton-pli/DySCO](https://github.com/princeton-pli/DySCO).
