---
title: "How to Break a Transformer's Loss Plateau (and Why Waiting Is a Lottery)"
date: 2026-10-22
draft: false
tags:
  - transformers
  - training-dynamics
  - in-context-learning
  - induction-heads
  - optimization
description: "A small transformer sits on a loss plateau for thousands of updates, and the same run repeated jumps at a different time. We show what the model builds while the loss is flat, how to test a fix when takeoff is a lottery, and which fixes cut the plateau."
aliases:
  - transformer-loss-plateau
  - mqar-loss-plateau
  - induction-head-plateau
---

Here are two runs of the same small transformer learning a simple recall task: remember which value went with a key, and give it back when the key returns. The runs are identical down to the seed and the order of the data. Both get stuck at the same low accuracy. One breaks free at update 14,000, under three minutes in; the other waits until update 21,750, just over four.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-lottery-dark.svg" alt="Accuracy against training updates for two runs of the same 4-layer transformer on MQAR with the same seed, data order and initial weights. Both climb to the copy level of about 0.10, the first by update 6,000 and the second by update 10,000, and stay flat; the first jumps toward 0.8 at update 14,000, the second at update 21,750." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-lottery-light.svg" alt="Accuracy against training updates for two runs of the same 4-layer transformer on MQAR with the same seed, data order and initial weights. Both climb to the copy level of about 0.10, the first by update 6,000 and the second by update 10,000, and stay flat; the first jumps toward 0.8 at update 14,000, the second at update 21,750." />

> [!note]- What the two runs share
> | Setting | Both runs |
> |---|---|
> | Task | multi-query associative recall (MQAR): 32-token rows of 15 key-value pairs, vocabulary of 256 tokens |
> | Model | 4 layers, width 128, 4 attention heads |
> | Training | AdamW, learning rate 1e-4, batch 64, evaluated every 250 updates on 512 fixed rows |
> | Kept the same | seed, initial weights, training rows and their order |
> | Left to differ | GPU and compiler nondeterminism (not isolated) |
> | Speed | about 87.6 updates per second on one RTX 4090, with the run alone on the card; every MQAR time in the post uses this rate |

The training loss tells the same story, only more quietly: it sits near 4.85 through the flat stretch and drops by about a quarter of a nat at takeoff. It drops so little because the loss averages over every token in a row, and most of them, such as a key seen for the first time, no model can predict.

On the task I care about, the graded language described below, the same lottery costs more. The original run took off after about 70,000 updates, about half an hour. A rerun from the same commit and seed tracked it that far and then never took off.

> [!note]- Graded timing
> The graded model (4 layers, width 128, 16 properties) runs at about 35.5 updates per second alone on the same RTX 4090, evaluation excluded. Every graded time in the post uses this rate.

What is the model doing while the loss is flat? And if even a rerun can miss takeoff, how do we test whether a fix helps? We will follow what I saw, tried and found, ending with a playbook for diagnosing a plateau and deciding what to change.

> [!summary] In short
> - [On the plateau](#what-has-the-model-already-learned-on-the-plateau) the model answers right about one time in $N$: it has learned to copy a value from the context, not which one.
> - [Takeoff is a lottery](#waiting-for-takeoff-is-a-lottery-and-the-tickets-are-expensive), so we [test a fix by branching](#can-we-test-a-fix-from-the-same-stuck-model) from one stuck checkpoint, with unchanged control branches beside it.
> - On MQAR, [repeated keys, Muon and a slow length ramp](#which-changes-shorten-a-transformers-loss-plateau) took off sooner than their controls; [noise and dropout](#can-noise-or-dropout-break-a-transformers-loss-plateau) did not.
> - [The plateau is not idle](#the-copy-path-improves-while-attention-stays-flat): handed the right position, the model copied its value better every checkpoint while its attention stayed flat.
> - On the graded language [the meaning match lagged instead](#on-the-graded-language-the-meaning-match-lags), and [Muon from scratch](#muon-learned-restatement-from-scratch-in-every-seed-of-one-recipe) learned it in every seed of one recipe.
> - [The playbook](#how-to-diagnose-and-break-a-transformers-loss-plateau): check accuracy against $1/N$, stop waiting, find the slow half, feed it, and test the fix against control branches.

## What has the model already learned on the plateau?

The recall task is **multi-query associative recall**, or MQAR ([Arora et al. 2023](https://arxiv.org/abs/2312.04927)). Here is an example row. Letters are keys; the token after each key is its value. A key keeps its value when it returns.

```text
A 3 B 7 C 1 A ...
```

We are predicting the token after the final `A`. We find its earlier copy at the start, then read the token after that copy: `3`. The intervening `B 7` and `C 1` belong to other keys. Copying either value would produce a token from the row, but the wrong answer. Solving the task requires both finding the matching key and copying what followed it. This is the operation of an **induction head** ([Olsson et al. 2022](https://arxiv.org/abs/2209.11895)), an attention head that uses an earlier occurrence to continue a repeated pattern. I use MQAR as the running example because it makes this lookup visible and a run takes off within 2 to 4 minutes.

The **graded language** is a small synthetic language. A document introduces entities, each as a few property words followed by a name, and mentions them again later; the model must produce the name. A **verbatim** mention repeats the introduction's words. A **restatement** describes the same properties in fresh words, so the model has to match on meaning, not on spelling. As an English stand-in (the real language uses made-up words): a document introduces "a tall, quiet painter called Mira", and later asks for "the lanky, soft-spoken artist", where the answer is "Mira". The graded language stands in for retrieval by meaning in large models, which is what my research is about. Its models also have 4 layers and a width of 128.

Both tasks plateau at the same kind of height. On the plateau the model has learned that the answer is one of the values, or names, already in the row, but not which one. If $N$ is the number of different candidates the row has shown before the question, an even guess among them is right one time in $N$:

$$
\text{accuracy on the plateau} \approx \frac{1}{N}.
$$

We call this the **copy level**. It is the dashed line in the figure, at about 0.1 on these MQAR rows.

The copy level is not chance. At the start of training both runs scored close to zero, below it, because they had not yet learned that the answer comes from the row. They climbed to the copy level and stayed there. So the flat stretch is a learned strategy: copy something from the context. It is cheap, it pays one time in $N$, and the loss has little more to say until the model learns *which* value to copy.

So the plateau has a floor the model built on purpose, and a jump that nobody can schedule. Before I could ask what the model builds on top of that floor, I had a practical problem to solve: if takeoff is a lottery, a fix tested on a fresh run proves nothing.

## Waiting for takeoff is a lottery, and the tickets are expensive

The pair in the figure was not unusual. Across eight seeds of the same MQAR model, takeoff came anywhere between update 11,750 and update 21,750, or between about two and four minutes. The gap between the earliest and the latest seed is nearly as long as the earliest seed's whole wait.

On the graded language the same lottery runs on a longer clock. Three seeds of one recipe took off more than 20,000 updates apart: one near update 40,000, one at 52,000 and only halfway, and one that was just beginning to rise when the run stopped at 60,000.

I had already spent about 17 runs this way over the preceding six research blocks. They ended at the copy level or below the accuracy I needed. Each next attempt bought more updates or a bigger model, then waited again. On one night, a run consumed about 3.6 of my 5.5 GPU hours and returned no usable result because it never left the plateau. I needed to make takeoff itself the question.

This is why a fix cannot earn credit by beating a baseline trained from another seed. Its apparent gain includes the lottery. Averaging that away would mean many seeds for each technique, each paying for a full run. I needed a comparison I could afford to repeat.

## Can we test a fix from the same stuck model?

We test a fix by **branching from a checkpoint**. Imagine your run is stuck and you want to change its training now. We copy one saved checkpoint from inside the plateau several times and keep training each copy, a **branch**. Some branches train as before; these are the **control branches**. The others switch the technique on: the **treated branches**. Comparing the two is a **branch test**.

To copy the method, train a baseline while saving checkpoints, then choose one during the flat loss. Mine was still below the copy level: the loss plateau had begun before the model learned to copy. Branch from it with several fresh data orders. For each order, run both a control branch and a treated branch. Count updates from the branch point until takeoff. Then turn the technique off and keep training normally, a **retention check** that the skill stays.

All branches inherit the same weights, so the lottery before the checkpoint is shared. The lottery after it is not: a fast treated branch on its own could simply be lucky again, and the control branches measure how much waiting still varies from that state. Evaluate every branch on ordinary rows, so a technique cannot pass just by making its own training examples easier.

> [!note]- The test's settings
> | Setting | Value |
> |---|---|
> | Checkpoint | MQAR seed 2 at update 5,500; its loss had been flat since update 1,000 and its original takeoff came at 16,500 |
> | Data orders | seeds 1, 2 and 3, the same for every arm |
> | Evaluation | every 250 updates on the same 512 ordinary rows |
> | Takeoff | the first evaluation above the midpoint between the copy level and 1.0 that stays above it |
> | Retention check | 11,000 ordinary updates with the technique off; every treated branch must finish at 0.95 accuracy or more |
> | Cost | about 124,000 updates for six branches, about 24 minutes at the solo rate |

The timing rule requires every treated branch to take off before every control branch. Why demand complete separation? If the technique does nothing, its three branches should have no special claim to the earliest three places among the six. There are twenty ways to choose those places, and only one puts all the treated branches first:

$$
P(\text{all three treated branches first by chance})
= \frac{1}{\binom{6}{3}} = \frac{1}{20} = 0.05.
$$

This is a permutation test at its most extreme outcome. It assumes no ties and that, under no effect, the treated and control labels could be swapped without changing anything. A pass also needs the retention check, and it speaks for the checkpoint it branched from.

The first technique to pass was **repetitions**: each key appeared about twice per row instead of about once. Here are its branches against the controls, all starting from the saved plateau checkpoint.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-resumes-dark.svg" alt="Accuracy against updates after branching from one MQAR checkpoint inside the plateau, three data orders per arm. The three control branches take off 9,000, 9,250 and 17,500 updates after the branch point; the three branches with repeated keys take off after 750, 1,000 and 2,500, all before the first control branch." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-resumes-light.svg" alt="Accuracy against updates after branching from one MQAR checkpoint inside the plateau, three data orders per arm. The three control branches take off 9,000, 9,250 and 17,500 updates after the branch point; the three branches with repeated keys take off after 750, 1,000 and 2,500, all before the first control branch." />

The slowest repetition branch took off after 2,500 updates, before the fastest control branch at 9,000. The medians were 1,000 against 9,250 updates, about nine times sooner: roughly 11 seconds against under two minutes at the rate above. After I turned repetitions off, all three still finished at 0.99 accuracy or more on ordinary rows.

## Which changes shorten a transformer's loss plateau?

I tested five levers: repeated keys, a different optimizer, a length curriculum, easy batches and sharper attention. Three passed: repetitions, Muon and a slow length ramp.

The figure compares all the levers with the control branches that ran beside them. Each grey bar belongs to its own test: even the same checkpoint and data order gave different control takeoff times in different tests. Borrowing a faster or slower reference from another test would change the verdict.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-screens-dark.svg" alt="Takeoff updates after branching, with one row per technique. Three dots show its branches; the grey bar behind them spans the control branches from the same test. Passing requires every dot to lie left of its bar and the skill to survive the retention check. Only the repeated-key rows, Muon at 3e-4 and 1e-4, and the ramp adding a pair every 750 updates clear their bars." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-screens-light.svg" alt="Takeoff updates after branching, with one row per technique. Three dots show its branches; the grey bar behind them spans the control branches from the same test. Passing requires every dot to lie left of its bar and the skill to survive the retention check. Only the repeated-key rows, Muon at 3e-4 and 1e-4, and the ramp adding a pair every 750 updates clear their bars." />

> [!note]- Reading the timings
> Minutes here use the same rate as above: 87.6 updates per second.
> For one data order, control takeoff ranged from 8,750 to 18,000 updates across tests. Rerunning an identical configuration moved takeoff by up to 1,500 updates, so small differences between doses are not readable.

### Repeated keys give the match more chances

Repetitions put another sighting of a key into the same length of row. Fewer distinct keys fit, but the model encounters the same association again:

```text
ordinary:  A 3 B 7 C 1 D 4 E 9 ... A ?
repeated:  A 3 B 7 A 3 C 1 B 7 ... A ?
```

[Zucchet et al. 2025a](https://arxiv.org/abs/2505.17863) found that repetitions speed recall learning and that more distinct competing items prolong the plateau. They also warned that too much repetition can teach a shortcut: answer with the most frequent value. In my test, repetitions passed the retention check, so that shortcut did not take hold. Every dose passed; the larger model passed too.

Each row now supplies more events that reinforce the right match, but it also holds fewer distinct keys. I did not separate the two.

> [!note]- Repetition doses and the larger model
> | Repetitions | Median takeoff, updates after branching |
> |---|---|
> | 1.5x | 2,500 |
> | 2x | 750 to 1,000 across tests |
> | 3x | 2,500 |
> | Control | 9,250 |
>
> All doses kept accuracy of at least 0.99 in the retention check. In the 4-layer, width-256 model, repetitions took off at 2,750 updates on every order. Its control branches took 24,750 and 28,500 updates; the third was still on the plateau at the 60,000-update cap.

### Muon changes how the optimizer spends a step

AdamW scales each coordinate of an update by its running size. **Muon** ([Jordan et al. 2024](https://kellerjordan.github.io/posts/muon/)) instead orthogonalises the update for a whole weight matrix, making its directions move at similar rates. A small direction that teaches the key match can then receive more of the step beside a large direction that already helps copy something.

That was the bet from [Gopalani and Hu 2025](https://arxiv.org/abs/2506.13688), who shortened a plateau in a linear model. Their learning rate also changed the step size. I therefore needed a control: could AdamW get the same benefit simply by taking larger steps?

Muon passed, taking off in under a minute. AdamW at the same learning rate fell below the copy level and stayed there. Both optimizers started fresh, so resetting AdamW's running averages cannot explain the difference. On MQAR, then, the gain comes from how Muon allocates the step, not from its size. I did not measure which direction receives the extra step.

> [!note]- Muon, fresh AdamW and the rate control
> Muon acted on hidden two-dimensional weight matrices, with five Newton-Schulz steps and `match_rms_adamw` scaling. Embeddings and norms used AdamW.
>
> | Arm in the learning-rate control | Takeoff updates, by data order |
> |---|---|
> | Muon, 3e-4 | 3,750; 5,000; 4,250 |
> | Fresh AdamW, 1e-4 | 11,000; 8,500; 15,250 |
> | Fresh AdamW, 3e-4 | No takeoff; final accuracy 0.03 to 0.06 |
>
> Muon at 1e-4 passed narrowly, taking 7,500 to 10,250 updates; the margin is within the rerun spread. At 1e-3 it took off early but failed retention: final accuracies ranged from 0.944 to 0.965, crossing below the 0.95 floor.

### Growing the row helps, within a narrow window

A **length curriculum** first asks the model to match among fewer keys, then adds more. We begin with a short row such as `A 3 B 7 A ?` and gradually introduce more intervening pairs.

[Garg et al. 2022](https://arxiv.org/abs/2208.01066) trained in-context learners with growing problems; [Zucchet et al. 2025b](https://arxiv.org/abs/2503.21676) found a faster start after warming up on a subset. The competition account also predicts a shorter plateau with fewer pairs.

The gradual ramp passed, taking off about a third sooner, in under a minute and a half. Evaluation was still on ordinary full-length rows, so this was transfer to the full task. Yet a slower ramp and a brief staged curriculum both failed. Giving the match an easier beginning helped only with the right time at each length.

> [!note]- Length schedules
> | Schedule | Takeoff updates after branching |
> |---|---|
> | Start at 8 tokens; add a pair every 750 updates | 6,500; 7,000; 7,250 |
> | Control beside that ramp | 10,250; 10,500; 18,000 |
> | Add a pair every 1,500 updates | 10,750 to 12,500; failed |
> | 8, 16, 24 tokens for 500 updates each | Failed |
>
> The passing ramp reached full length, 32 tokens, at update 9,000 after branching. Takeoff happened before full length.

### Mixing in easy batches did not transfer

Easy batches leave ordinary training in place and mix in simpler examples. I tried short rows and a **fixed map**: on those rows, `A` always means `3`, so the weights can memorise the answer without reading the context.

[Kim et al. 2024](https://arxiv.org/abs/2410.05448) shortened an in-context plateau by mixing tasks. [Kirsch et al. 2022](https://arxiv.org/abs/2212.04458) eliminated a plateau with enough fixed-map examples. Here neither recipe passed. A small share of short rows did not help; a large share harmed learning. The fixed map harmed both shares.

My guess at why the published recipe fails here: a memorised answer competes with the context, since on ordinary rows `A` can carry another value, and that trains against the lookup the task needs.

> [!note]- Easy-batch shares
> | Easy examples | Share | Result |
> |---|---|---|
> | Short rows, 8 tokens | 10% | Takeoff 6,000 to 9,500; control 9,000 to 13,500; failed separation |
> | Short rows, 8 tokens | 90% | Two branches never took off; final accuracies 0.447 and 0.297 |
> | Fixed map | 10% | Harmed learning |
> | Fixed map | 90% | Accuracy on ordinary rows fell to 0.01 |

### Sharper attention tests two opposing accounts

**Sharpening attention** increases the differences between its scores before softmax. A head spreading its attention across the row then concentrates more on whichever positions already score highest. I applied that change at the branch point and gradually removed it, a treatment often called **heat treatment**.

[Hoffmann et al. 2023](https://arxiv.org/abs/2310.12956) traced stalls to attention staying near uniform and found sharpening could help. [Zucchet et al. 2025a](https://arxiv.org/abs/2505.17863) predict no expected benefit while the scores are random: making a guess more confident does not make it correct more often. These accounts disagree about what the stalled model lacks.

No sharpening factor helped. Takeoff took about 2 to 4 minutes. Attention did become less uniform immediately, so the intervention reached its target. The failure fits Zucchet's prediction; Hoffmann's result came from different tasks. Sharpening alone did not teach the model which key to choose.

> [!note]- Sharpening settings and takeoff
> I multiplied attention logits after QK-norm by 4, 12 or 24, annealing linearly back to 1 over 4,500 updates. Attention entropy fell at the first update.
> Takeoff ranged from 11,000 to 22,000 updates, against the controls' 9,000 to 13,500. No factor passed.

### Considered and set aside

- **Grokking fixes**, such as [Lee et al. 2024](https://arxiv.org/abs/2405.20233): grokking follows fitting the training data. Here the training loss itself is still flat.
- **Bigger batches** ([Aoyama et al. 2025](https://arxiv.org/abs/2511.16893)): fewer updates can cost more samples. I kept batch size fixed rather than treating that exchange as a cure.
- **Larger initial attention weights:** QK-norm largely removes their scale; changing initialisation also cannot help an existing checkpoint.
- **Mimetic initialisation** ([Trockman and Kolter 2023](https://arxiv.org/abs/2305.09828)): starts query-key products near identity, but applies only from scratch.
- **Supervising attention:** MQAR labels the right key, but real data lacks that answer key. I parked it for diagnosing a slow match.
- **Adding QK-norm** ([Dehghani et al. 2023](https://arxiv.org/abs/2302.05442)): guards against diverging attention scores; the model here already has it.
- **Other optimizer and attention changes:** equalising query, key and value gradients, learning a temperature, zeroing initial queries and changing Adam's beta2 lacked supporting plateau evidence at this scale or had failed in prior tests.

The passes suggest a shared account: more matching events, a fairer share of the step, or an easier match first. All change the gradient's size or direction, not its randomness. But could the model instead be stuck and need noise or dropout to explore its way out?

## Can noise or dropout break a transformer's loss plateau?

I thought the model might need to explore. It spreads attention evenly, collects its $1/N$, and has no reason to try another pattern. Sharpening amplifies the preferences it already has. Noise could create a preference it has never tried.

This was a reasonable account. Perturbed gradient descent escapes saddle points faster than plain gradient descent ([Jin et al. 2017](https://arxiv.org/abs/1703.00887)). Shrink-and-perturb pulls a trained network toward a fresh one to restore its ability to learn ([Ash and Adams 2020](https://arxiv.org/abs/1910.08475)). I tested four ways to disturb the stuck model.

### Four ways to shake the model

- **Weight noise:** after every update, add a little Gaussian noise to every hidden weight matrix, about as large as one update's step. It fades out over 6,000 updates. One arm kept it constant.
- **Attention-score noise:** jitter the attention scores before the softmax on every training row, so the model's picks are reshuffled a little each time. This aims at the match directly.
- **Attention dropout:** drop a random tenth of the attention weights on each training forward.
- **Shrink-and-perturb, once:** before the first update, every weight becomes 0.6 times itself plus 0.1 times a freshly initialised weight.

### The disturbances worked; the fixes did not

Each disturbance reached its target: score noise made training attention less uniform, and shrinking cut the hidden matrices to about 60 percent of their size.

> [!note]- Manipulation checks
> Attention entropy, a measure of how evenly attention spreads, fell from 1.90 to 1.58 with score noise at 0.5 and to 1.12 at 1.0. Shrink-and-perturb reduced the hidden matrices' typical size from 0.0194 to 0.0118.

None passed the branch test. The closest arm averaged 12,417 updates to takeoff against the controls' 13,000, within the rerun spread. Constant noise did no better than fading noise, so removing it too early does not explain the failure. Most arms also made takeoff less predictable.

> [!note]- Noise test: takeoff updates after branching
> Every arm, the control included, started with fresh AdamW. Each cell lists the three data orders. Weight noise is a fraction of each matrix's typical value; score noise is a multiple of the scores' own spread.
>
> | Arm | Takeoff | Passed |
> |---|---|---|
> | Control | 11,250; 11,750; 16,000 | reference |
> | Weight noise 1e-3, fading | 11,000; 18,750; 11,000 | no |
> | Weight noise 3e-3, fading | 12,000; 11,750; 19,250 | no |
> | Weight noise 1e-3, constant | 12,250; 11,250; 13,750 | no |
> | Score noise 0.5, fading | 17,000; 11,750; 16,750 | no |
> | Score noise 1.0, fading | 20,250; 13,000; 10,750 | no |
> | Attention dropout 0.1 | 17,000; 13,500; 13,000 | no |
> | Shrink-and-perturb | 18,250; 13,500; 22,250 | no |
>
> Fresh-AdamW reruns moved takeoff by up to 3,250 updates. No gain exceeded that spread. With score noise at 1.0, two of three branches ended below 0.95 accuracy in the retention check. Not tested: larger or longer noise, noise on the query-key product alone, and noise beside Muon.

Shrinking was later on every data order, costing about 5,000 updates on average, about a minute. If the plateau were idle, there would be little to undo. The delay suggested that something had been accumulating while the loss stayed flat.

## The copy path improves while attention stays flat

The two halves are **find the match** and **copy its value**. The copy path runs through the head's value and output weights and the later layers. We can read them separately.

### Attention finds the right value only near takeoff

We first read the answer row's attention to the correct value, relative to an average other position. On a log scale, zero means no preference. Across all eight baseline seeds, this stayed flat through most of the plateau and rose near takeoff.

> [!note]- Reading the stored attention evaluations
> No new training was needed. Evaluations were stored every 250 updates. During the first half of the plateau, attention moved by -4 to +11 percent of its eventual rise. The last 2,000 updates before takeoff carried a median of about 0.87 of that rise.

By this measurement the model still looks stuck. But attention reads only the match.

### Force the match to read the copy path

We can hand the model the right position and ask whether it can read out the value. At each saved checkpoint, we force the answer row's attention in one layer onto the correct value positions. Every other position is blocked; softmax redistributes attention over the permitted ones. We then read the answer's **lead**: its logit, or output score, minus the mean logit of the row's other values. With matching done for it, that lead measures the copy path's readiness.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-forced-read-dark.svg" alt="Schematic of the forced read on the row A 3 B 7 C 1 A ?. Ordinary: the answer row spreads its attention evenly over the earlier tokens, and its scores for the values 3, 7 and 1 are equal, so the lead is zero. Forced: all attention goes to the 3, every other position is blocked, and the score for 3 rises above 7 and 1; the gap is the lead." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-forced-read-light.svg" alt="Schematic of the forced read on the row A 3 B 7 C 1 A ?. Ordinary: the answer row spreads its attention evenly over the earlier tokens, and its scores for the values 3, 7 and 1 are equal, so the lead is zero. Forced: all attention goes to the 3, every other position is blocked, and the score for 3 rises above 7 and 1; the gap is the lead." />

The forced lead rose steadily in every seed, from about one logit at plateau start to two or three by its middle. Without forcing, the lead stayed near zero. Both rose at takeoff. The model was learning to copy whatever it attended to while still attending to nothing in particular.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-inside-dark.svg" alt="Two stacked panels show all eight seeds as thin lines, sharing an axis from plateau start to takeoff. Top: attention to the correct value stays flat until a late rise near takeoff. Bottom: the answer's lead with attention forced onto the correct value rises steadily from the start of the plateau in every seed, while the unforced lead stays near zero." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-inside-light.svg" alt="Two stacked panels show all eight seeds as thin lines, sharing an axis from plateau start to takeoff. Top: attention to the correct value stays flat until a late rise near takeoff. Bottom: the answer's lead with attention forced onto the correct value rises steadily from the start of the plateau in every seed, while the unforced lead stays near zero." />

> [!note]- Forced lead on saved checkpoints
> Checkpoints were read every 500 updates. In the first half of the plateau, rank correlation between forced lead and updates was 0.98 to 1.00. The ordinary lead stayed between 0.01 and 0.08 through the plateau.
>
> | Seed | Plateau start | Takeoff | Forced lead at start | At midpoint | At takeoff |
> |---|---|---|---|---|---|
> | 1 | 9,250 | 21,750 | 1.0 | 2.8 | 8.7 |
> | 2 | 10,500 | 16,500 | 1.4 | 2.2 | 7.1 |
> | 3 | 5,750 | 11,750 | 0.9 | 1.8 | 4.3 |
> | 4 | 6,250 | 15,750 | 0.8 | 2.3 | 6.4 |
> | 5 | 6,750 | 19,500 | 1.1 | 3.1 | 8.9 |
> | 6 | 9,000 | 20,750 | 1.1 | 2.4 | 3.2 |
> | 7 | 5,750 | 14,500 | 1.0 | 2.3 | 8.1 |
> | 8 | 3,750 | 13,750 | 0.9 | 2.6 | 7.8 |

### The forced read follows the value we supply

Forcing attention onto a wrong value made the model favour that wrong value. Forcing it onto keys gave the target nothing. Dividing the lead by the spread of value logits preserved its rise, so growing logits alone cannot explain it. These controls held in every seed.

> [!note]- Controls and their limit
> The wrong value gained 0.92 to 0.95 of the forced lead. Forcing keys gained at most 0.022 target logits, against forced leads of 1.8 to 3.1. The normalised lead rose in all eight seeds.
>
> Forcing every head in a layer is off distribution. All controls share that method.

### Two weak factors can hide steady learning

This explains why noise failed. Gradient descent was already building the copy path, and random jitter spreads across all directions, with little pointing toward the match. Shrinking cut every weight, the copy path's included, which fits the 5,000 updates it cost.

The loss hides this progress because copying pays only when the match selects the right value. The match, in turn, gets a useful signal only when copying can deliver that value. Attention can snap into place once copying pays.

Each half's gradient is scaled by the other's strength. Two weak halves barely move; improvement in either accelerates the other. This gives the slow start and sudden rise of deep linear networks ([Saxe et al. 2014](https://arxiv.org/abs/1312.6120)). We can picture the halves as factors $a$ and $b$. For their product $u=ab$ starting small at $u_0$ and growing toward $s$,

$$
t \approx \frac{\tau}{s}\ln\!\left(\frac{s}{u_0}\right).
$$

Here $\tau$ sets the learning timescale. A bigger start shortens the wait only logarithmically; a faster rate shortens it in proportion. Starting 100 times larger buys about as much as learning twice as fast:

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-two-factors-dark.svg" alt="Three S-shaped curves of the product u = ab against time, from the formula above. The baseline stays near zero for a long time and rises around time 9. A 100 times larger start and a twice faster rate both rise around time 4.6; the faster rate is steeper." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-two-factors-light.svg" alt="Three S-shaped curves of the product u = ab against time, from the formula above. The baseline stays near zero for a long time and rises around time 9. A 100 times larger start and a twice faster rate both rise around time 4.6; the faster rate is steeper." />

In this picture the three levers that passed all act on the rate: repetitions supply more matching events, Muon gives a small direction more of the step, and the ramp begins with an easier match.

### Copying first is a known order, not a universal rule

[Bietti et al. 2023](https://arxiv.org/abs/2306.00802) found the output matrix learning before the query-key match. [Singh et al. 2024](https://arxiv.org/abs/2404.07129) also found copying quicker to learn. I rediscovered that order with a practical read on saved transformer checkpoints. But [Gopalani and Hu 2025](https://arxiv.org/abs/2506.13688) identify attention as the bottleneck in their setting. The task matters.

> [!note]- Related orders in small models
> [Zucchet et al. 2025a](https://arxiv.org/abs/2505.17863) found feedforward learning before relevant attention, followed by positive feedback. [Reddy 2023](https://arxiv.org/abs/2312.03002) found a slow contextual-label phase before induction attention appeared at the cliff. [Musat et al. 2025](https://arxiv.org/abs/2511.01033) found output learning before query-key learning in a minimal model. My read uses the standard 4-layer transformer.

### Which half sets the clock is still open

I tried to act on each half by branching from the stuck checkpoint. A higher learning rate on either half did not help; ten times the rate on the copy half broke training. Handing the model one half from later in the same run roughly halved the wait, and the match half did nearly as well as the copy half. So both halves had moved by then. The order is measured. That the copy half alone sets the clock is not.

> [!note]- Acting on one half
> Copy half means the value and output weights; match half means the query and key weights. "From update 13,000" swaps in that half's weights from the same run at update 13,000, where the forced lead was about 2 logits and attention was still flat.
>
> | Arm | Takeoff updates after branching, by data order | End accuracy |
> |---|---|---|
> | Control | 13,500; 11,500; 13,250 | 1.00 |
> | Copy half at 10x learning rate | never | 0.11 to 0.12 |
> | Copy half at 3x | 13,750; 25,250; 19,750 | 0.96 to 1.00 |
> | Match half at 10x | never; 19,500; 21,750 | 0.13 to 1.00 |
> | Copy half from update 13,000 | 5,750; 6,500; 5,500 | 0.96 to 1.00 |
> | Match half from update 13,000 | 6,750; 7,750; 7,250 | 0.97 to 0.98 |
> | All weights from update 13,000 | 3,500; 3,750; 3,750 | 0.94 to 0.97 |
>
> The planned causal reading was void: the all-weights control ended at 0.941 on one order, so these swaps are descriptive.

We now have two reads that need no new training: attention tells us whether matching improves; the forced lead tells us whether copying improves. On MQAR the copy path moved first. On the graded language, which half is slow?

## On the graded language, the meaning match lags

Back to the graded language from section 1, where "a tall, quiet painter called Mira" is later asked for as "the lanky, soft-spoken artist". The plateau I care about is on restatements, after the model can already answer verbatim mentions.

My recipe trains verbatim alone before mixing in restatements. The stuck model has therefore had a chance to learn the lookup with the original description before the recipe asks it to connect different words.

### More mentions did not shorten this plateau

Repeating keys cut the MQAR plateau about ninefold. Its first graded analogue was to mention each entity more often in a document. If seeing an association again was enough, this should have helped too.

It did nothing. Restatement accuracy stayed where it started, level with the control branches. The extra mentions made documents longer, so each update also cost more. Muon was the only lever in this test that moved restatement: every branch rose from about 0.50 to about 0.74. It rescued a partial skill, though it did not finish the task at this model size.

> [!note]- More mentions and Muon from the same checkpoint
> The test branched from an AdamW baseline at update 60,000, with restatement accuracy about 0.49. Each arm used three data orders and a 12,000-update cap. Mentions doubled from 12 to 24 per document.
>
> | Arm | Final pooled restatement accuracy, by order | Cost per update |
> |---|---|---|
> | Control | 0.506; 0.517; 0.510 | 1x |
> | Doubled mentions | 0.503; 0.498; 0.495 | 1.7x |
> | Muon, 3e-4 | 0.738; 0.746; 0.741 | 1x |
>
> Muon's rise came 8,000 to 10,000 updates after branching. At many entities, both mention forms fell together, consistent with a capacity limit at this size.

### The copy path was already ready

I ran the same two reads on the graded baselines' saved checkpoints. Here we force attention onto the positions holding the target entity's name. In the example, we hand the model "Mira", then compare its lead with the lead under ordinary attention.

The forced lead was already at least 2 logits at every plateau checkpoint in every seed. The ordinary lead stayed near 0.9. Handed the right entity, the model could copy its name. Left to choose, it did not find that entity. The missing skill was the meaning match.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-graded-halves-dark.svg" alt="On the graded language's restatement plateau, three AdamW runs have a forced lead of at least 2 logits when attention is forced onto the target entity's name, while the unforced lead stays near 0.9. On one run the forced lead climbs past 6, well above the 1 to 3 logits at which MQAR took off, while restatement accuracy stays at the copy level." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-graded-halves-light.svg" alt="On the graded language's restatement plateau, three AdamW runs have a forced lead of at least 2 logits when attention is forced onto the target entity's name, while the unforced lead stays near 0.9. On one run the forced lead climbs past 6, well above the 1 to 3 logits at which MQAR took off, while restatement accuracy stays at the copy level." />

The clearest run kept improving its forced lead without improving restatement accuracy. On MQAR, attention had snapped into place after a much smaller forced lead. Here a stronger copy path kept waiting for a match. A forced lead is a read of readiness, not a universal takeoff threshold. Here it says the answer gets through once the choice is made, so the next intervention belongs on the match.

> [!note]- The graded checkpoints and the registered verdict
> These reads used checkpoints every 4,000 updates from the three AdamW 1e-4 baselines, with no new training. The lead is the target name's logit minus the mean logit of the other names earlier in the document.
>
> | Seed | Forced lead on restatements | Ordinary lead | Restatement accuracy |
> |---|---|---|---|
> | 1 | Near 3 through the plateau | Near 0.9 | Copy level through 52,000, then rising |
> | 2 | Climbed from 3.6 to 6.7 | 0.83 to 0.92 | 0.38 to 0.39 through 56,000 |
> | 3 | 2.4 to 2.7 through 36,000; then rose with takeoff | 0.77 to 0.79 before takeoff | Took off at 40,000 to 44,000 |
>
> Seed 2 had verbatim accuracy of 0.99 to 1.00 from update 32,000. On MQAR, takeoff came with a forced lead of about 1 to 3 logits.
>
> The registered verdict was MIXED: the rule required both eligible seeds to show "match lags" at update 36,000, but seed 3's restatement forced lead was under half its still-learning verbatim lead. The descriptive reading across the three seeds is that the match lagged.

### Repeating an entity does not repeat its meaning link

Every ordinary restatement draws fresh words: another mention of Mira can use "long-legged" where the last used "lanky". So doubled mentions repeated the name, which the model could already copy, and not the link that makes "lanky" match "tall".

### Repeating the word-to-property link did not take off either

The aimed version reuses words. Within one document, every restatement of a property uses the first fresh word the document drew for it:

```text
ordinary:  lanky ... long-legged ... towering      (three words for "tall")
reuse:     lanky ... lanky ... lanky               (one word, three sightings)
```

A new document draws its own words. Length, vocabulary and answers stay the same. A companion arm instead makes restatements a larger share of the mentions. Each arm branches from a checkpoint where verbatim is solved and the forced lead is strong, so a gain has to come from the remaining restatement skill.

If repetition works by feeding the slow half, reuse should pass the branch test and the retention check. It did not. The control branches took off after about 14,000 updates, and the denser arm did no better. Reuse made things worse: none of its branches took off, and restatement drifted down from 0.39 to about 0.32. Every branch that did take off, the controls included, stopped at a second level near 0.77 instead of finishing.

Why would reuse hurt? One possible reason: when a property always gets the same word, a later restatement can be answered from an earlier restatement of that word. That route stays inside the document and never needs the word-to-property map that fresh words ask for. Fewer distinct words and more sightings of each also changed together, so a plain loss of variety is not ruled out. On the graded language, Muon remained the only lever that moved restatement beyond its control branches.

> [!note]- Aimed-repetition setup and results
> The branches start from an AdamW baseline at update 48,000: restatement 0.39, verbatim 1.00, forced lead 6.5. Each arm has three data orders, a 20,000-update cap, then a 5,000-update retention check. Takeoff is the first lasting crossing of 0.695, halfway from 0.39 to 1.0, read every 1,000 updates.
>
> | Arm | Takeoff updates, by data order | Restatement after the retention check | Distinct restatement words per document |
> |---|---|---|---|
> | Control | 15,000; 14,000; 14,000 | 0.77 to 0.78 | about 51 |
> | Denser restatements | 13,000; 13,000; 15,000 | 0.77 to 0.78 | about 77 |
> | Reused words | none by 20,000 | 0.32 to 0.33 | 14 |
>
> The denser arm put restatements in 0.76 to 0.78 of mentions, against half for the control. Reuse left token counts, answers and the loss unchanged. No arm could have met the 0.95 end bar in this window: every branch that took off levelled near 0.77.

## Muon learned restatement from scratch in every seed of one recipe

### A rescue has to become a recipe

Rescuing a stuck run does not tell us how to start the next one. I therefore tested Muon from the first update, keeping the graded recipe used by the AdamW baselines.

With the original AdamW recipe, only one of three seeds learned restatement, after about 48,000 updates. The others had not learned it by 60,000. One stayed at the copy level throughout. The other could answer verbatim mentions but still guessed among names on restatements.

With Muon, all three seeds reached at least 0.99 restatement accuracy at the first evaluation after the warm stage. They also finished at that level when the document contained many more entities. The improvement began before restatements arrived: Muon learned the verbatim task earlier too.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-muon-scratch-dark.svg" alt="Restatement accuracy against updates for nine graded runs trained from scratch. During the 20,000-update verbatim-only stage, restatement falls below the copy level. All three Muon runs jump to about 1.0 at the first check after that stage. Two of three AdamW 2.5e-4 runs do the same. The AdamW 1e-4 runs mostly stay at the copy level, with one rising near 44,000." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-muon-scratch-light.svg" alt="Restatement accuracy against updates for nine graded runs trained from scratch. During the 20,000-update verbatim-only stage, restatement falls below the copy level. All three Muon runs jump to about 1.0 at the first check after that stage. Two of three AdamW 2.5e-4 runs do the same. The AdamW 1e-4 runs mostly stay at the copy level, with one rising near 44,000." />

The dip in the shaded stage is expected. While only verbatim is trained, a restatement is answered with whichever entity's words match best, which is usually the wrong one.

Counting the stalled AdamW runs, Muon used about eight times less training time per learned model under the same conditions. Its updates cost about the same at this size.

> [!note]- The from-scratch adoption test
> All arms used 4 layers, width 128, 16 properties and batch 64. The first 20,000 updates used verbatim mentions only; later training mixed verbatim and restatement equally. There were three seeds per arm.
>
> | Recipe | Seeds that learned restatement | First learned evaluation | Final restatement at 32 to 64 entities |
> |---|---|---|---|
> | AdamW, 1e-4 | 1 of 3 | About 48,000 updates | 0.99 for that seed; others at the copy level |
> | AdamW, 2.5e-4 | 2 of 3 | 22,000 updates | 0.995 to 0.999; seed 1 partial at 0.64 |
> | Muon, 3e-4 | 3 of 3 | 22,000 updates | 0.996 to 0.999 |
>
> Every Muon seed reached at least 0.96 verbatim accuracy by update 8,000. AdamW 1e-4 was at 0.64 to 0.66 or at the copy level then. Updates per second were about the same for both optimizers.
>
> Higher rates do not keep helping. In a graded branch test without weight decay, Muon at 1e-3 and 3e-3 doubled and quadrupled the weight norms within 5,000 updates and left the plateau in place, so watch the weight norm when raising the rate.

### A matched AdamW step nearly reproduced the result

Muon took larger steps through the hidden matrices than the original AdamW recipe, so the control raised AdamW's rate until its measured movement per update matched Muon's. That larger AdamW step learned the same way in two of three seeds. The third failed to finish verbatim during the warm stage, took off late, and ended with partial restatement skill.

So on the graded language, three seeds do not separate Muon from a larger step, unlike MQAR, where AdamW at Muon's learning rate collapsed below the copy level. I adopted Muon anyway: it learned in every seed and costs nothing extra per update.

> [!note]- Matching the movement of the weights
> Muon 3e-4 moved each hidden matrix about 2.4 times as far per update as AdamW 1e-4. AdamW 2.5e-4 matched it: both moved about 5e-5 of the weights per update.
>
> The larger-step AdamW seed 1 stayed at the copy level through update 32,000. Its final pooled restatement accuracy was 0.77; the 0.64 in the table is for 32 to 64 entities.

### Finishing verbatim predicted a quick meaning match, post hoc

Every run that had fully learned verbatim when restatements arrived learned restatement within 2,000 updates. The original AdamW runs entered the mixed stage with verbatim still unfinished and stalled. This is a post hoc observation, not a registered test.

This fits the product picture from section 4: with the identity match and copy path finished, a new meaning match pays as soon as it improves, so the last factor gets a strong gradient from its first update. It suggests advancing the curriculum when verbatim is learned, not after a fixed number of updates.

> [!note]- The warm-stage observation
> All three Muon seeds and the two fast AdamW 2.5e-4 seeds had verbatim accuracy of at least 0.99 when restatements joined. The AdamW 1e-4 seeds stood at 0.73 to 0.89. The quick restatement gain came within the next 2,000 updates.

### On nearby settings the plateau came back

Muon did not remove the plateau everywhere. On the same 16-property language, three more Muon seeds learned restatement again, 2,000 to 6,000 updates after restatements arrived.

Then I changed one thing at a time. In one change, each head's query-key product, the part that scores every earlier position against the question, was limited to half its usual rank. In the other, the language had 4 properties instead of 16. Seven such runs sat at about 0.3 restatement for 8,000 updates or more. Only two left, late. Verbatim held in every run.

So Muon gave a reliable recipe for one model and one language. It is not a cure for the plateau, and a new setting needs the reads again.

> [!note]- The narrower read and the 4-property language
> Every run used Muon 3e-4, 4 layers and width 128, a 10,000-update verbatim stage, then 14,000 mixed updates. The query-key product was factored in every run, at rank 32 (the full head width) or 16.
>
> | Language and rank | Runs that left the restatement plateau by 24,000 | Restatement at 4 to 7 entities, 24,000 |
> |---|---|---|
> | 16 properties, rank 32 | 3 of 3, within 6,000 mixed updates | 0.997 to 0.999 |
> | 16 properties, rank 16 | 1 of 3, near 20,000, still rising | 0.34; 0.87; 0.34 |
> | 4 properties, rank 32 | 1 of 2, near 22,000 | 0.34; 0.99 |
> | 4 properties, rank 16 | 0 of 2 | 0.31; 0.33 |
>
> The 4-property language uses the same 6,912 words but 576 carrier words instead of 144, so it does not isolate the property count. Rank and the factor's effective step size are not separated either.

## How to diagnose and break a transformer's loss plateau

Here is what I would do on the next stuck run, in order.

<img class="theme-dark-only figure-center" src="/blog/assets/plateau-v-playbook-dark.svg" alt="A five-step flow diagram of the playbook: accuracy flat at 1/N, then save checkpoints and stop waiting, then two reads to find which half is flat, then feed the slow half, then test the fix against control branches." />
<img class="theme-light-only figure-center" src="/blog/assets/plateau-v-playbook-light.svg" alt="A five-step flow diagram of the playbook: accuracy flat at 1/N, then save checkpoints and stop waiting, then two reads to find which half is flat, then feed the slow half, then test the fix against control branches." />

1. **Check accuracy against $1/N$, not just the loss.** Count the candidates a row has shown before the question. Accuracy at one over that count, with a flat loss, is this plateau: the model copies something from the context, just not the right thing. The loss can hide it; on MQAR it moved by a quarter of a nat at takeoff (section 1).
2. **Stop waiting.** Takeoff is a lottery; identical reruns differed by thousands of updates, and one never took off (section 2). Set an early check for the recipe you run, and stop a run that misses it.
3. **Find the slow half on saved checkpoints.** Read attention to the right position for the match, and the forced lead for the copy half. Whichever stays flat is slow. Neither read needs new training (sections 4 and 5).
4. **Test fixes by branching from a checkpoint.** Branch from one checkpoint inside the plateau on three data orders, with control branches beside every arm. Pass a fix only if all its branches take off before all the controls, then turn it off and check the skill stays (section 2).
5. **Feed the slow half.** On MQAR, repeat what it learns from: the keys. On the graded language, neither more mentions nor reused words moved restatement. Try Muon; it passed on MQAR and made every from-scratch run of my graded recipe learn, though the plateau returned on nearby settings. A slow length ramp helped MQAR, in a narrow window (sections 3, 5 and 6).
6. **Finish the easy version first.** Runs that had finished verbatim added the meaning match quickly (section 6, post hoc).

> [!note]- The early check I use for graded Muon runs
> Require verbatim accuracy of at least 0.9 by update 10,000. This check belongs to my graded recipe; it is not a threshold measured for other tasks or model sizes.

### Do not use these as fixes for this plateau

These failed or harmed learning in my tests, at the tested settings:

- Weight noise, attention-score noise, attention dropout and shrink-and-perturb: none passed (section 4).
- Attention sharpening or heat treatment: attention changed, but takeoff did not improve (section 3).
- Easy-batch mixing with shorter rows or a fixed key-to-value map: neither passed; the fixed map harmed learning (section 3).
- A short staged curriculum: it failed where a slower ramp passed (section 3).
- A higher AdamW rate on a stuck MQAR run, or a higher rate on one circuit half: these broke or delayed learning (sections 3 and 4).
- Repetition on the graded language: doubled mentions added cost without moving restatement, and reusing restatement words did not take off (section 5).

### Model size did not transfer as a cure

Model size moved the plateau in opposite directions. A wider graded model learned restatement where the narrow one did not. A wider MQAR model waited longer: its control branches needed 24,750 updates, 28,500, and more than 60,000, against 9,000 to 17,500 for the narrow model in section 2. Size is a setting to record, not a lever to reach for.

> [!note]- The size comparison
> The wider graded model (width 256) also had a larger MLP and was read on one seed, so width and MLP changed together. Each MQAR model branched from its own plateau checkpoint.

Batch size stayed fixed. [Aoyama et al. 2025](https://arxiv.org/abs/2511.16893) make it a knob to record when comparing plateaus, not a cure: fewer updates are useful only after counting the samples they consume.

## Stop buying tickets

The title said waiting is a lottery. What replaced the lottery was not a smarter wait. It was two cheap reads that say which half of the lookup is slow, and a fair test for the fix that feeds it.

On the graded language, a recipe that needed a lucky seed and about 48,000 updates gave way to one that learned at the first check after the warm stage, in every seed I ran on that model and language. One step away, with a narrower query-key product or another language, the plateau came back. That is why the reads come first: they say which half to feed on the run in front of you. We can test what breaks the plateau before paying to wait through it again.

## Notes on how this was made

I designed, ran and reviewed these experiments with the assistance of AI agents (Claude and Codex). The figures were drawn by Codex from the stored run data. The training code builds on a small research library that is not public yet.

**References.**

- Aoyama et al., "Predicting the Emergence of Induction Heads in Language Model Pretraining," [arXiv 2511.16893](https://arxiv.org/abs/2511.16893).
- Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," [arXiv 2312.04927](https://arxiv.org/abs/2312.04927).
- Ash, Adams, "On Warm-Starting Neural Network Training," [arXiv 1910.08475](https://arxiv.org/abs/1910.08475).
- Bietti et al., "Birth of a Transformer: A Memory Viewpoint," [arXiv 2306.00802](https://arxiv.org/abs/2306.00802).
- Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters," [arXiv 2302.05442](https://arxiv.org/abs/2302.05442).
- Garg et al., "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes," [arXiv 2208.01066](https://arxiv.org/abs/2208.01066).
- Gopalani, Hu, "What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers," [arXiv 2506.13688](https://arxiv.org/abs/2506.13688).
- Hoffmann et al., "Eureka-Moments in Transformers: Multi-Step Tasks Reveal Softmax Induced Optimization Problems," [arXiv 2310.12956](https://arxiv.org/abs/2310.12956).
- Jin et al., "How to Escape Saddle Points Efficiently," [arXiv 1703.00887](https://arxiv.org/abs/1703.00887).
- Jordan et al., "Muon: An optimizer for hidden layers in neural networks," [blog post, 2024](https://kellerjordan.github.io/posts/muon/).
- Kim et al., "Task Diversity Shortens the ICL Plateau," [arXiv 2410.05448](https://arxiv.org/abs/2410.05448).
- Kirsch et al., "General-Purpose In-Context Learning by Meta-Learning Transformers," [arXiv 2212.04458](https://arxiv.org/abs/2212.04458).
- Lee et al., "Grokfast: Accelerated Grokking by Amplifying Slow Gradients," [arXiv 2405.20233](https://arxiv.org/abs/2405.20233).
- Musat et al., "On the Emergence of Induction Heads for In-Context Learning," [arXiv 2511.01033](https://arxiv.org/abs/2511.01033).
- Olsson et al., "In-context Learning and Induction Heads," [arXiv 2209.11895](https://arxiv.org/abs/2209.11895).
- Reddy, "The mechanistic basis of data dependence and abrupt learning in an in-context classification task," [arXiv 2312.03002](https://arxiv.org/abs/2312.03002).
- Saxe et al., "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks," [arXiv 1312.6120](https://arxiv.org/abs/1312.6120).
- Singh et al., "What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation," [arXiv 2404.07129](https://arxiv.org/abs/2404.07129).
- Trockman, Kolter, "Mimetic Initialization of Self-Attention Layers," [arXiv 2305.09828](https://arxiv.org/abs/2305.09828).
- Zucchet et al., "The emergence of sparse attention: impact of data distribution and benefits of repetition," [arXiv 2505.17863](https://arxiv.org/abs/2505.17863).
- Zucchet et al., "How do language models learn facts? Dynamics, curricula and hallucinations," [arXiv 2503.21676](https://arxiv.org/abs/2503.21676).
