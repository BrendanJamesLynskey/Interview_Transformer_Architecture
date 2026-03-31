# Worked Problem 02 — Compute-Optimal Scaling with Chinchilla Laws

## Background

The **Chinchilla scaling laws** (Hoffmann et al., 2022, "Training Compute-Optimal
Large Language Models") established that prior large language models were
significantly undertrained. Given a fixed compute budget $C$ (in FLOPs), there
is an optimal allocation of parameters $N^*$ and training tokens $D^*$ that
minimises the loss.

This problem works through the quantitative relationships and applies them to
GPT-3 as a concrete benchmark.

---

## Key Relationships

### The compute approximation

The dominant cost of training a Transformer is the matrix multiplications in the
forward and backward passes. A widely-used approximation is:

$$
C \approx 6ND
$$

where:
- $C$ is the total training compute in FLOPs
- $N$ is the number of model parameters
- $D$ is the number of training tokens

The factor of 6 arises from: 2 FLOPs per multiply-add $\times$ 3 passes (forward,
backward weights, backward activations). This is an approximation that ignores
attention FLOPs and embedding layers, which are typically small relative to the
feed-forward and projection layers for large models.

### Chinchilla optimal allocation

The Chinchilla paper derived (via empirical fitting over many training runs) that
the loss-optimal allocation satisfies:

$$
N^* \propto C^{0.5}, \qquad D^* \propto C^{0.5}
$$

More precisely, the optimal ratio of tokens to parameters is approximately:

$$
\frac{D^*}{N^*} \approx 20
$$

That is, **each parameter should see approximately 20 training tokens**.

Combined with $C = 6ND$, this gives:

$$
N^* = \sqrt{\frac{C}{6 \times 20}} = \sqrt{\frac{C}{120}}, \qquad D^* = 20 N^*
$$

---

## Part A — Given a Compute Budget, Derive Optimal N and D

### Problem

You have a compute budget of $C = 2.4 \times 10^{23}$ FLOPs (approximately
the budget used to train GPT-3, rounded for clean arithmetic). Find:

1. The compute-optimal number of parameters $N^*$
2. The compute-optimal number of training tokens $D^*$
3. Verification that $C \approx 6 N^* D^*$

### Solution

**Step 1 — Solve for $N^*$.**

$$
N^* = \sqrt{\frac{C}{120}} = \sqrt{\frac{2.4 \times 10^{23}}{120}}
= \sqrt{2.0 \times 10^{21}}
$$

$$
N^* = \sqrt{2.0} \times 10^{10.5} = 1.414 \times 10^{10.5}
$$

$$
10^{10.5} = 10^{10} \times 10^{0.5} = 10^{10} \times 3.162 = 3.162 \times 10^{10}
$$

$$
N^* = 1.414 \times 3.162 \times 10^{10} \approx 4.47 \times 10^{10} \approx \boxed{44.7\text{ billion parameters}}
$$

**Step 2 — Solve for $D^*$.**

$$
D^* = 20 \times N^* = 20 \times 4.47 \times 10^{10} \approx 8.94 \times 10^{11} \approx \boxed{894\text{ billion tokens}}
$$

**Step 3 — Verify compute consistency.**

$$
6 N^* D^* = 6 \times 4.47 \times 10^{10} \times 8.94 \times 10^{11}
= 6 \times 3.995 \times 10^{22}
= 2.397 \times 10^{23} \approx C \checkmark
$$

---

## Part B — GPT-3 Was Undertrained: Quantitative Analysis

### GPT-3 Configuration

- Parameters: $N_{\text{GPT-3}} = 175 \times 10^9 = 1.75 \times 10^{11}$
- Training tokens: $D_{\text{GPT-3}} = 300 \times 10^9 = 3.0 \times 10^{11}$

### Step 1 — Compute GPT-3's actual compute budget

$$
C_{\text{GPT-3}} = 6 N_{\text{GPT-3}} D_{\text{GPT-3}}
= 6 \times 1.75 \times 10^{11} \times 3.0 \times 10^{11}
= 6 \times 5.25 \times 10^{22}
= 3.15 \times 10^{23} \text{ FLOPs}
$$

### Step 2 — Compute Chinchilla-optimal N and D for $C_{\text{GPT-3}}$

$$
N^*_{\text{for GPT-3 budget}} = \sqrt{\frac{3.15 \times 10^{23}}{120}}
= \sqrt{2.625 \times 10^{21}}
= 1.620 \times 10^{10.5}
\approx 5.12 \times 10^{10}
\approx \boxed{51.2\text{ billion parameters}}
$$

$$
D^*_{\text{for GPT-3 budget}} = 20 \times 5.12 \times 10^{10} \approx \boxed{1.02\text{ trillion tokens}}
$$

### Step 3 — Comparison table

| Quantity | GPT-3 actual | Chinchilla optimal (same $C$) | Ratio |
|----------|-------------|-------------------------------|-------|
| Parameters $N$ | 175B | 51B | GPT-3 is $3.4\times$ over-parameterised |
| Training tokens $D$ | 300B | 1,020B | GPT-3 sees $3.4\times$ too few tokens |
| $D/N$ ratio | 1.71 | 20 | GPT-3 ratio is $11.7\times$ below optimal |

### Step 4 — Interpretation

GPT-3 used its compute budget to build a very large model and then trained it for
relatively few token steps. The Chinchilla analysis shows that for the same
compute budget you could train a $51$B parameter model on $1$T tokens and achieve
**lower loss** (better performance) than the $175$B parameter GPT-3.

This is not merely theoretical: **Chinchilla 70B** (trained on 1.4T tokens) was
shown to match or exceed GPT-3 on most benchmarks while being $2.5\times$ smaller.
**LLaMA-1 7B** (trained on 1T tokens) was similarly competitive with much larger
models trained on fewer tokens.

---

## Part C — Generalisation: Solving for $N^*$ and $D^*$ from Any Budget

The general procedure for any compute budget $C$:

$$
\boxed{N^* = \sqrt{\frac{C}{120}}, \qquad D^* = 20 N^* = \sqrt{\frac{C}{3}}}
$$

Let us tabulate this for a range of budgets:

| Compute $C$ (FLOPs) | $N^*$ (params) | $D^*$ (tokens) | Example |
|--------------------|----------------|----------------|---------|
| $10^{19}$ | 289M | 5.8B | Small research model |
| $10^{21}$ | 2.9B | 57.7B | Mid-scale model |
| $10^{22}$ | 9.1B | 182B | — |
| $3.15 \times 10^{23}$ | 51B | 1.02T | GPT-3 budget, Chinchilla-optimal |
| $10^{24}$ | 91B | 1.83T | — |
| $10^{25}$ | 289B | 5.77T | Large frontier model (2023–2024) |

**Computing each row:** $N^* = \sqrt{C / 120}$, $D^* = 20 N^*$.

---

## Part D — Caveats and Limitations

The Chinchilla $D/N \approx 20$ ratio is a guideline, not a hard law:

1. **Inference cost matters.** If a model will be deployed for billions of
   queries, it may be worth *over-training* a smaller model (more tokens than
   Chinchilla optimal) to reduce per-query inference cost. LLaMA-1 and Mistral
   models were explicitly designed with this trade-off: train longer to get a
   smaller, faster model.

2. **Data quality and diversity.** The Chinchilla fits were performed on a
   specific data mixture. High-quality data (code, mathematics, curated web) can
   yield lower loss per token, shifting the effective optimum.

3. **Repetition of data.** Once the dataset is exhausted, training on repeated
   data exhibits diminishing returns. The token budget $D^*$ assumes unique
   (or near-unique) tokens.

4. **Beyond next-token prediction.** Chinchilla laws are derived from language
   modelling loss. Downstream task performance (MMLU, HumanEval, etc.) does not
   always track loss linearly, particularly for emergent capabilities that appear
   only above threshold model sizes.

5. **The $C \approx 6ND$ approximation.** This ignores the quadratic cost of
   attention over sequence length and embedding tables. For very long sequences
   or large vocabularies, the true compute is somewhat higher.

---

## Summary

| Key formula | Expression |
|-------------|------------|
| Compute budget | $C \approx 6ND$ |
| Optimal parameters | $N^* = \sqrt{C / 120}$ |
| Optimal tokens | $D^* = 20 N^*$ |
| Optimal $D/N$ ratio | $\approx 20$ |
| GPT-3 $D/N$ ratio | $\approx 1.71$ (undertrained by $\sim 12\times$) |

The central lesson: **scaling parameters and scaling data are equally important.
A model half the size, trained on twice the data, will often outperform the
larger model on the same compute budget.**
