# Worked Problem 03 — Effective Batch Size, Gradient Accumulation, and Critical Batch Size

## Problem Statement

A team is training a 7B parameter language model. They have access to:

- 8 GPUs, each with 80 GB HBM
- Per-GPU micro-batch size: $B_\text{micro} = 4$ sequences
- Sequence length: $L = 2048$ tokens
- Gradient accumulation steps: $k = 16$

An experiment has estimated the **gradient noise scale** (Mccandlish et al., 2018)
for this model-dataset combination at the start of training:

$$
\mathcal{B}_\text{noise} = \frac{\text{tr}(\Sigma)}{\|\bar{g}\|^2} \approx 500{,}000 \text{ tokens}
$$

**Tasks:**

1. Compute the effective batch size in sequences and in tokens.
2. State and apply the linear scaling rule for learning rate.
3. Determine whether the team is operating in the noise-limited or compute-limited
   regime, and quantify the efficiency loss.
4. Find the compute-optimal batch size that maximises useful work per FLOP.
5. Show how to read the critical batch size from a training efficiency curve.

---

## Part 1 — Effective Batch Size

### Sequences per update step

With gradient accumulation, each GPU performs $k$ forward-backward passes before
the optimiser update. The effective batch size per GPU is:

$$
B_\text{eff,GPU} = k \times B_\text{micro} = 16 \times 4 = 64 \text{ sequences}
$$

Across all 8 GPUs (data parallelism, gradients are all-reduced before the update):

$$
B_\text{eff} = N_\text{GPU} \times k \times B_\text{micro} = 8 \times 16 \times 4 = \boxed{512 \text{ sequences}}
$$

### Tokens per update step

Each sequence has $L = 2048$ tokens:

$$
B_\text{tokens} = B_\text{eff} \times L = 512 \times 2048 = \boxed{1{,}048{,}576 \approx 1.05 \times 10^6 \text{ tokens}}
$$

This is approximately 1M tokens per gradient update — a common target for
large language model training.

---

## Part 2 — Linear Scaling Rule for Learning Rate

**The linear scaling rule** (Goyal et al., 2017) states that when multiplying
the batch size by a factor $m$ relative to a reference batch size
$B_\text{ref}$, the learning rate should be scaled by the same factor:

$$
\eta_\text{new} = \eta_\text{ref} \times \frac{B_\text{eff}}{B_\text{ref}}
$$

**Why this holds (intuition).** With a batch $m$ times larger, each gradient
update is $m$ times more accurate (lower variance). To cover the same distance
in weight space in fewer steps, each step must be $m$ times larger — hence the
$m\times$ increase in learning rate.

**Practical application.** Suppose prior work established that $\eta_\text{ref} =
3 \times 10^{-4}$ works well with $B_\text{ref} = 256$ sequences (262,144 tokens).
The team's effective batch is $512$ sequences, so:

$$
\eta_\text{new} = 3 \times 10^{-4} \times \frac{512}{256} = 3 \times 10^{-4} \times 2 = 6 \times 10^{-4}
$$

**Caveat.** The linear scaling rule begins to break down for very large batches
(well above $B_\text{crit}$), where a square-root scaling rule can be more
appropriate:

$$
\eta_\text{new} = \eta_\text{ref} \times \sqrt{\frac{B_\text{eff}}{B_\text{ref}}}
$$

The two rules agree near $B_\text{crit}$; at large multiples of $B_\text{crit}$,
increasing the LR sub-linearly is safer.

---

## Part 3 — Noise-Limited vs Compute-Limited Regime

### The critical batch size

The **critical batch size** $B_\text{crit}$ is the batch size (in tokens) at
which noise and signal contribute equally to gradient quality. It is approximately
equal to the gradient noise scale:

$$
B_\text{crit} \approx \mathcal{B}_\text{noise} = 500{,}000 \text{ tokens}
$$

The team's batch size is $B_\text{tokens} = 1{,}048{,}576 \approx 1.05 \times 10^6$
tokens.

### Regime determination

$$
\frac{B_\text{tokens}}{B_\text{crit}} = \frac{1.05 \times 10^6}{5 \times 10^5} \approx 2.1
$$

The team is operating at approximately **$2.1\times$ the critical batch size**,
meaning they are in the **compute-limited regime** (also called the
"batch-saturated" regime).

### Quantifying efficiency loss

The relationship between batch size, training steps, and data processed is:

$$
S(B) = S_\min \cdot \left(1 + \frac{B_\text{crit}}{B}\right), \qquad
E(B) = E_\min \cdot \left(1 + \frac{B}{B_\text{crit}}\right)
$$

where $S(B)$ is the number of steps to reach a target loss and $E(B)$ is the
total tokens (data) consumed to reach the same loss. $S_\min$ and $E_\min$ are
the theoretical minimum steps and data respectively.

**Steps efficiency** — how many steps does the team need relative to minimum?

$$
\frac{S(B)}{S_\min} = 1 + \frac{B_\text{crit}}{B} = 1 + \frac{1}{2.1} \approx 1.48
$$

The team takes $1.48\times$ more steps than necessary to reach a given loss,
meaning $\sim 32\%$ of compute per step goes to gradient noise reduction they
are past the saturation point for.

**Data efficiency** — how much data does the team consume relative to minimum?

$$
\frac{E(B)}{E_\min} = 1 + \frac{B}{B_\text{crit}} = 1 + 2.1 = 3.1
$$

The team consumes $3.1\times$ more tokens than the theoretical minimum. Each
individual batch update is using many more tokens than necessary to achieve its
incremental loss reduction.

### Summary of regime trade-offs

| Batch size | Steps to target | Tokens to target | Practical concern |
|-----------|----------------|-----------------|-------------------|
| $B \ll B_\text{crit}$ | Many (noisy gradients) | Near-minimum | Very slow wall-clock time |
| $B = B_\text{crit}$ | $2 S_\min$ | $2 E_\min$ | Balanced |
| $B = 2 B_\text{crit}$ | $1.5 S_\min$ | $3 E_\min$ | Team's situation |
| $B = 10 B_\text{crit}$ | $1.1 S_\min$ | $11 E_\min$ | Diminishing step returns |

The team is trading **data efficiency** for **step efficiency** (fewer, faster
steps). This is usually the right trade-off when GPU utilisation is the
bottleneck and data is not scarce, but it means the model may need to be
exposed to more unique data tokens than a smaller-batch regime.

---

## Part 4 — Compute-Optimal Batch Size

A useful heuristic for the **compute-optimal batch size** — the one that
minimises total FLOPs to a target loss — is:

$$
B_\text{opt} = B_\text{crit}
$$

At $B = B_\text{crit}$, both steps and data overhead are minimised jointly (each
is $2\times$ the theoretical minimum, which is the Pareto-optimal point). Any
deviation from $B_\text{crit}$ increases *either* step count *or* data
consumption multiplicatively.

For the team's model:

$$
B_\text{opt} = B_\text{crit} = 500{,}000 \text{ tokens}
$$

$$
B_\text{opt,sequences} = \frac{500{,}000}{2048} \approx 244 \text{ sequences}
$$

To achieve 244 sequences with 8 GPUs and micro-batch 4:

$$
k_\text{opt} = \frac{B_\text{opt,sequences}}{N_\text{GPU} \times B_\text{micro}}
= \frac{244}{8 \times 4} = \frac{244}{32} \approx 8 \text{ accumulation steps}
$$

**Recommendation:** Reduce gradient accumulation from 16 to 8 steps. This halves
the effective batch size to $\approx 524{,}288$ tokens, close to $B_\text{crit}$.

---

## Part 5 — Reading $B_\text{crit}$ from a Training Efficiency Curve

In practice, $B_\text{noise}$ (and thus $B_\text{crit}$) is estimated
empirically. The procedure:

**Protocol.** Train multiple runs with identical hyperparameters except for batch
size $B \in \{B_1, B_2, \ldots, B_n\}$. For each run, measure the number of
optimiser steps $S(B)$ to reach a fixed reference loss $\mathcal{L}_\text{target}$.

**Fitting.** Plot $1/S(B)$ against $B$. From the formula above:

$$
\frac{1}{S(B)} = \frac{1}{S_\min} \cdot \frac{1}{1 + B_\text{crit}/B}
= \frac{1}{S_\min} \cdot \frac{B}{B + B_\text{crit}}
$$

Rearranging:

$$
\frac{S_\min}{S(B)} = \frac{B}{B + B_\text{crit}}
\implies \frac{1}{S(B)} = \frac{1}{S_\min} \cdot \frac{1}{1 + B_\text{crit}/B}
$$

In linear form: if we plot $S_\min / S(B)$ vs $1/B$, we get a line with slope
$B_\text{crit}/S_\min$ and intercept $1/S_\min$. A linear regression on
$(1/B,\; S_\min/S(B))$ immediately yields both $S_\min$ and $B_\text{crit}$.

**Alternatively**, the simpler two-point estimate:

$$
B_\text{crit} \approx \frac{S(B_\text{small}) \cdot B_\text{small} - S(B_\text{large}) \cdot B_\text{large}}{S(B_\text{large}) - S(B_\text{small})}
$$

where $B_\text{small}$ and $B_\text{large}$ are two batch sizes with
$B_\text{small} \ll B_\text{crit} \ll B_\text{large}$ (if known approximately
in advance).

---

## Summary Table

| Quantity | Value | Formula |
|----------|-------|---------|
| Micro-batch size | 4 sequences / GPU | given |
| GPUs | 8 | given |
| Gradient accumulation steps | 16 | given |
| Effective batch (sequences) | 512 | $N_\text{GPU} \times k \times B_\text{micro}$ |
| Effective batch (tokens) | 1,048,576 | $B_\text{eff} \times L$ |
| Critical batch size | 500,000 tokens | given (measured) |
| $B / B_\text{crit}$ ratio | 2.1 | compute-limited regime |
| Step overhead vs minimum | $1.48\times$ | $1 + B_\text{crit}/B$ |
| Data overhead vs minimum | $3.1\times$ | $1 + B/B_\text{crit}$ |
| Optimal $k$ (accumulation) | $\approx 8$ | $B_\text{crit} / (N_\text{GPU} \times B_\text{micro} \times L)$ |
