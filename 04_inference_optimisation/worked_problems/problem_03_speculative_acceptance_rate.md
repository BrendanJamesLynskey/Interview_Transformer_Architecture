# Worked Problem 03 — Speculative Decoding: Acceptance Rate and Expected Speedup

## Problem Statement

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) uses a small
**draft model** to generate candidate tokens, which a large **target model** then
verifies in a single parallel forward pass. This achieves target-model quality at
sub-target-model latency.

**Tasks:**

1. State the acceptance criterion for a single draft token.
2. Prove that the acceptance-rejection sampling procedure produces exact samples
   from the target distribution $p$.
3. Derive the expected number of accepted tokens per speculative step.
4. Derive the expected wall-clock speedup formula.
5. Work through a numerical example with given distributions.

---

## Background

**Setup.**
- Target model: distribution $p(x \mid \text{context})$ over vocabulary, slow to
  evaluate (latency $\tau_p$ per forward pass).
- Draft model: distribution $q(x \mid \text{context})$ over vocabulary, fast
  to evaluate (latency $\tau_q \ll \tau_p$).
- Speculation length: $\gamma$ draft tokens are generated per speculative step.

**Goal.** Generate tokens with the exact distribution $p$ while spending less
time than running $p$ autoregressively.

---

## Part 1 — The Acceptance Criterion

For each draft token $\tilde{x}$ sampled from $q(\cdot)$, the target model
computes $p(\tilde{x})$. The token is **accepted** with probability:

$$
\alpha(\tilde{x}) = \min\!\left(1,\; \frac{p(\tilde{x})}{q(\tilde{x})}\right)
$$

**Intuition.** If the target model assigns at least as much probability to $\tilde{x}$
as the draft model does ($p(\tilde{x}) \geq q(\tilde{x})$), accept always — the
draft was at least as conservative as the target. If the draft overestimates
the probability ($q(\tilde{x}) > p(\tilde{x})$), accept with probability
$p(\tilde{x}) / q(\tilde{x}) < 1$.

This is a form of **rejection sampling** adapted to work token-by-token.

---

## Part 2 — Proof of Correctness (Exact Sampling from $p$)

We must show that the token actually produced at any position has distribution
$p$.

**Case 1: Token is accepted.** Draft token $\tilde{x}$ is sampled from $q$ and
accepted with probability $\alpha(\tilde{x})$. The probability of producing token
$x$ by this path is:

$$
P(\text{accept},\; X = x) = q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right)
= \min(q(x),\; p(x))
$$

**Case 2: Token is rejected and resampled.** When $\tilde{x}$ is rejected, we
do not output $\tilde{x}$. Instead, we sample a replacement token from the
**adjusted distribution**:

$$
p'(x) = \frac{\max(0,\; p(x) - q(x))}{\sum_{x'} \max(0,\; p(x') - q(x'))}
$$

First, compute the normalisation constant. The total probability mass of the
accepted path is:

$$
\sum_x \min(q(x), p(x)) = 1 - \sum_x \max(0, q(x) - p(x))
$$

(Since $\sum_x p(x) = \sum_x q(x) = 1$, the "excess" mass of $q$ over $p$
equals the "deficit" of $q$ under $p$.) Let $\beta = \sum_x \max(0, q(x) - p(x))$.

The probability of rejection on any given draft token $\tilde{x}$ is:

$$
P(\text{reject}) = \sum_{\tilde{x}} q(\tilde{x})\left(1 - \min\!\left(1, \frac{p(\tilde{x})}{q(\tilde{x})}\right)\right)
= \sum_{\tilde{x}} \max(0, q(\tilde{x}) - p(\tilde{x})) = \beta
$$

The probability of producing token $x$ via the rejection path is:

$$
P(\text{reject},\; X = x) = \beta \cdot p'(x) = \beta \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'}\max(0, p(x') - q(x'))} = \max(0, p(x) - q(x))
$$

**Total probability of producing $x$:**

$$
P(X = x) = \min(q(x), p(x)) + \max(0, p(x) - q(x))
$$

For any $x$, exactly one of two cases holds:
- $p(x) \geq q(x)$: $\min = q(x)$, $\max = p(x) - q(x)$. Sum $= q(x) + p(x) - q(x) = p(x)$.
- $p(x) < q(x)$: $\min = p(x)$, $\max = 0$. Sum $= p(x) + 0 = p(x)$.

In both cases, $P(X = x) = p(x)$. $\blacksquare$

The procedure produces exact samples from the target distribution $p$, regardless
of how poorly calibrated the draft model $q$ is. A bad draft only wastes compute
by increasing rejections; it never produces off-distribution tokens.

---

## Part 3 — Expected Number of Accepted Tokens per Step

**Single-token acceptance probability.** For a single draft token, the probability
of acceptance is:

$$
\alpha = \sum_x q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right)
= \sum_x \min(q(x), p(x))
= 1 - \frac{1}{2}\sum_x |p(x) - q(x)|
= 1 - \frac{d_\text{TV}(p, q)}{1}
$$

where $d_\text{TV}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ is the
**total variation distance** between the draft and target distributions. So:

$$
\boxed{\alpha = 1 - d_\text{TV}(p, q)}
$$

A draft model close to the target in TV distance gives high acceptance rates.

**With $\gamma$ draft tokens.** Draft tokens are generated autoregressively, but
their acceptance events are approximately independent if the context is similar.
Under this independence assumption:

The expected number of accepted draft tokens $\mathbb{E}[A]$ follows from the
geometric-like process: token $k$ is accepted only if tokens $1, \ldots, k-1$
were all accepted. Thus:

$$
P(\text{exactly } k \text{ tokens accepted, } k < \gamma) = \alpha^k (1 - \alpha)
$$

$$
P(\text{all } \gamma \text{ tokens accepted}) = \alpha^\gamma
$$

The expected number of accepted tokens (not counting the bonus token from
resampling on rejection) is:

$$
\mathbb{E}[A] = \sum_{k=0}^{\gamma-1} k \cdot \alpha^k (1-\alpha) + \gamma \cdot \alpha^\gamma
$$

Using the geometric series result:

$$
\mathbb{E}[A] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} - 1 \cdot (\text{adjustment for the bonus token})
$$

More precisely, in speculative decoding **one token is always generated per step**
(either the last accepted draft token, the resampled rejection token, or the bonus
token after full acceptance). The total expected tokens produced per speculative
round is:

$$
\mathbb{E}[\text{tokens per round}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

**Derivation.** Let $N$ be the number of tokens produced. $N = k+1$ if the first
$k$ drafts are accepted and the $(k+1)$-th is rejected (or $k = \gamma$ and the
bonus token is generated):

$$
\mathbb{E}[N] = \sum_{k=0}^{\gamma} (k+1)\, P(\text{first } k \text{ accepted, stop at } k+1)
$$

$$
= \sum_{k=0}^{\gamma-1} (k+1) \alpha^k (1-\alpha) + (\gamma + 1) \alpha^\gamma
$$

Let $S = \sum_{k=0}^{\gamma} (k+1) \alpha^k (1-\alpha)$ (geometric partial sum identity):

$$
\mathbb{E}[N] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

This can be verified by summing the series directly:

$$
\sum_{k=0}^{\gamma} (k+1) \alpha^k = \frac{d}{d\alpha}\sum_{k=0}^{\gamma} \alpha^{k+1}
= \frac{d}{d\alpha} \frac{\alpha(1-\alpha^{\gamma+1})}{1-\alpha}
$$

which evaluates to $\dfrac{1 - (\gamma+2)\alpha^{\gamma+1} + (\gamma+1)\alpha^{\gamma+2}}{(1-\alpha)^2}$,
and after multiplying by $(1-\alpha)$ and simplifying yields $\dfrac{1 - \alpha^{\gamma+1}}{1 - \alpha}$.

---

## Part 4 — Expected Speedup Formula

**Without speculative decoding.** Generating $T$ tokens requires $T$ sequential
target model calls:

$$
\text{Time}_\text{baseline} = T \cdot \tau_p
$$

**With speculative decoding.** Each round costs:
- $\gamma$ serial draft model calls: $\gamma \tau_q$
- 1 parallel target model call (verifies all $\gamma$ tokens simultaneously): $\tau_p$

Each round produces $\mathbb{E}[N] = \dfrac{1 - \alpha^{\gamma+1}}{1 - \alpha}$ tokens.

Expected time per token:

$$
\bar{t} = \frac{\gamma \tau_q + \tau_p}{\mathbb{E}[N]} = \frac{\gamma \tau_q + \tau_p}{\dfrac{1 - \alpha^{\gamma+1}}{1 - \alpha}}
$$

**Speedup** relative to non-speculative:

$$
\boxed{
S = \frac{\tau_p}{\bar{t}} = \frac{\tau_p (1 - \alpha^{\gamma+1})}{(1 - \alpha)(\gamma \tau_q + \tau_p)}
}
$$

**Simplified for small $\tau_q / \tau_p$** (draft model is cheap). Let
$c = \tau_q / \tau_p \ll 1$:

$$
S \approx \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(1 + \gamma c)}
$$

For $c \to 0$ (free draft model):

$$
S_\text{max} = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} = \mathbb{E}[N]
$$

The maximum possible speedup equals the expected number of tokens per round —
which makes intuitive sense.

---

## Part 5 — Numerical Example

### Setup

Vocabulary: 5 tokens. A single context position with:

$$
p = [0.50,\; 0.25,\; 0.15,\; 0.07,\; 0.03]
\qquad \text{(target model)}
$$

$$
q = [0.35,\; 0.30,\; 0.20,\; 0.10,\; 0.05]
\qquad \text{(draft model)}
$$

Speculation length: $\gamma = 4$.

Latency ratio: $\tau_q / \tau_p = 0.1$ (draft model is $10\times$ faster).

### Step 1: Compute per-token acceptance probabilities

$$
\alpha_i = \min\!\left(1, \frac{p_i}{q_i}\right)
$$

| Token | $p_i$ | $q_i$ | $p_i/q_i$ | $\alpha_i$ |
|-------|-------|-------|------------|------------|
| 0 | 0.50 | 0.35 | 1.429 | 1.000 |
| 1 | 0.25 | 0.30 | 0.833 | 0.833 |
| 2 | 0.15 | 0.20 | 0.750 | 0.750 |
| 3 | 0.07 | 0.10 | 0.700 | 0.700 |
| 4 | 0.03 | 0.05 | 0.600 | 0.600 |

### Step 2: Compute overall acceptance rate $\alpha$

$$
\alpha = \sum_i q_i \cdot \alpha_i
= 0.35(1.0) + 0.30(0.833) + 0.20(0.750) + 0.10(0.700) + 0.05(0.600)
$$
$$
= 0.350 + 0.250 + 0.150 + 0.070 + 0.030 = 0.850
$$

**Verification via TV distance:**

$$
d_\text{TV}(p, q) = \frac{1}{2}\sum_i |p_i - q_i|
= \frac{1}{2}(|0.50 - 0.35| + |0.25 - 0.30| + |0.15 - 0.20| + |0.07 - 0.10| + |0.03 - 0.05|)
$$
$$
= \frac{1}{2}(0.15 + 0.05 + 0.05 + 0.03 + 0.02) = \frac{1}{2}(0.30) = 0.15
$$

$$
\alpha = 1 - d_\text{TV} = 1 - 0.15 = 0.850 \checkmark
$$

### Step 3: Expected tokens per round

$$
\mathbb{E}[N] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
= \frac{1 - (0.85)^5}{1 - 0.85}
= \frac{1 - 0.4437}{0.15}
= \frac{0.5563}{0.15}
= 3.709 \text{ tokens per round}
$$

### Step 4: Expected speedup

$$
S = \frac{\tau_p (1 - \alpha^{\gamma+1})}{(1 - \alpha)(\gamma \tau_q + \tau_p)}
= \frac{1 \cdot 0.5563}{0.15 \times (4 \times 0.1 + 1)}
= \frac{0.5563}{0.15 \times 1.4}
= \frac{0.5563}{0.210}
= \mathbf{2.65\times}
$$

With a free draft model ($c = 0$):

$$
S_\text{max} = \mathbb{E}[N] = 3.71\times
$$

The $10\%$ overhead of the draft model reduces the ideal speedup from $3.71\times$
to $2.65\times$ — the draft model cost is non-negligible even at $10\times$ faster.

### Step 5: Sensitivity to $\gamma$

| $\gamma$ | $\mathbb{E}[N]$ | $S$ (with $c=0.1$) |
|----------|----------------|---------------------|
| 1 | $1.85$ | $1.85 / (0.1 + 1) = 1.68\times$ |
| 2 | $2.55$ | $2.55 / (0.2 + 1) = 2.13\times$ |
| 4 | $3.71$ | $3.71 / (0.4 + 1) = 2.65\times$ |
| 8 | $5.47$ | $5.47 / (0.8 + 1) = 3.04\times$ |
| 16 | $6.52$ | $6.52 / (1.6 + 1) = 2.51\times$ |
| 32 | $6.66$ | $6.66 / (3.2 + 1) = 1.59\times$ |

**Observation.** There is an optimal $\gamma$ that maximises speedup. Too small
and the speculation is wasted; too large and the draft model overhead dominates.
The optimal $\gamma$ satisfies:

$$
\frac{dS}{d\gamma} = 0 \implies \gamma_\text{opt} \approx \frac{\log(1 - (1-\alpha)/\ln(1/\alpha))}{\log \alpha}
$$

In practice, $\gamma = 4$–$8$ is typical for a draft model that is $10\times$
faster and achieves $\alpha \approx 0.7$–$0.9$.

---

## Part 6 — Practical Considerations

**Draft model selection.** The draft model is typically the same architecture
at smaller scale (e.g., 68M or 160M parameters drafting for a 7B model), or
a purpose-trained distilled model. The same tokeniser and vocabulary must be
used — acceptance probabilities are only well-defined when $p$ and $q$ are over
the same set of tokens.

**Context-dependence of $\alpha$.** The acceptance rate $\alpha$ is not constant
— it depends on the context. On factual prompts with peaked distributions
(both $p$ and $q$ concentrate on a few tokens), $\alpha$ is high. On open-ended
creative prompts with flat distributions, $\alpha$ is lower. Real systems report
$\alpha \approx 0.6$–$0.9$ on typical benchmarks.

**Batch inference.** Speculative decoding is most beneficial for a single request
(batch size 1). In large-batch serving, the target model is already compute-bound
(not memory-bandwidth-bound), so the decode step does not become faster with
fewer requests — and speculative decoding can actually reduce throughput because
it wastes tokens when drafts are rejected. Continuous batching and speculative
decoding address orthogonal bottlenecks.

**Self-speculative decoding (Medusa, EAGLE, Draft-and-Verify).** An alternative
to a separate draft model is to add extra heads to the target model itself
that predict tokens at future positions in parallel, eliminating the need for
a separate smaller model. EAGLE-2 (Li et al., 2024) achieves $3$–$4\times$
speedup this way on open-source models.

---

## Summary

| Quantity | Formula |
|----------|---------|
| Acceptance probability (single token) | $\alpha = 1 - d_\text{TV}(p, q) = \sum_x \min(p(x), q(x))$ |
| Expected tokens per round | $\mathbb{E}[N] = \dfrac{1 - \alpha^{\gamma+1}}{1 - \alpha}$ |
| Speedup | $S = \dfrac{\tau_p \cdot \mathbb{E}[N]}{\gamma \tau_q + \tau_p}$ |
| Numerical example ($\alpha=0.85$, $\gamma=4$, $c=0.1$) | $S = 2.65\times$ |
