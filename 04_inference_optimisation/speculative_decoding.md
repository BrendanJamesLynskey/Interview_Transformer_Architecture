# Speculative Decoding

## Overview

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) uses a small draft model to generate candidate tokens, which are then verified in parallel by the larger target model. When candidates are accepted, multiple tokens are produced per target model forward pass, improving throughput without changing output quality.

---

## Fundamentals

### Q1. What problem does speculative decoding solve?

**Answer.**

Autoregressive decoding is sequential by nature: each token requires a separate forward pass through the full model. The model is memory-bandwidth-limited during decode, so:

- A 70B parameter model in FP16 reads $\sim 140$ GB of weights per decode step
- On an A100 with 2 TB/s bandwidth, this takes $\sim 70$ ms per token — regardless of batch size
- This gives a hard limit of $\sim 14$ tokens/second per sequence

The fundamental bottleneck is that each token requires one sequential memory access over all model weights. You cannot parallelise across tokens during standard decode.

**Speculative decoding's insight.** The target model (large, slow) is good but slow. A draft model (small, fast) can guess what the target would produce. If the draft guesses correctly, verifying $k$ tokens costs the same as one target model step (because verification is parallel), effectively producing $k$ tokens per target model step.

**Result:** Up to $k$ tokens per target model forward pass, where $k$ depends on how often the draft model's guesses match the target.

---

### Q2. How does the speculative decoding algorithm work step by step?

**Answer.**

**Setup:** Target model $T$ (large, slow), draft model $D$ (small, fast), lookahead length $K$ (draft tokens per round).

**One round of speculative decoding:**

1. **Draft phase.** Run the draft model autoregressively for $K$ steps, producing:
   - Draft tokens: $\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_K$
   - Draft probabilities: $q(\hat{x}_i \mid x_{<i})$ for each draft step

2. **Verify phase.** Run the target model in a **single forward pass** on the sequence $[x_{<0}, \hat{x}_1, \hat{x}_2, \ldots, \hat{x}_K]$:
   - The target model processes all $K+1$ positions simultaneously (using causal masking)
   - This produces target probabilities $p(x \mid x_{<i})$ for all $K+1$ positions

3. **Accept/reject.** For each draft token $\hat{x}_i$ in order ($i = 1, 2, \ldots, K$):
   - With probability $\min\!\left(1, \frac{p(\hat{x}_i)}{q(\hat{x}_i)}\right)$: **accept** $\hat{x}_i$
   - Otherwise: **reject**, sample a replacement from a corrected distribution (see Q4), and stop

4. **Bonus token.** After accepting all $K$ tokens (or if the round completed without rejection), sample an additional token from $p(\cdot \mid x_{<K+1})$ — the target model's output at the last position.

5. **Repeat.** Use the accepted tokens as context and run the next round.

**Key guarantee:** The acceptance/rejection procedure ensures the output token distribution is exactly $p(\cdot)$ — the target model's distribution — regardless of draft model quality. This is not an approximation.

---

### Q3. Why does the acceptance/rejection procedure preserve the target distribution?

**Answer.**

The acceptance rule ensures that for any token $x$, the probability of accepting $x$ as the output equals $p(x)$.

**Theorem.** The procedure produces samples from $p$ (the target distribution).

**Proof sketch.** For a given draft token $\hat{x}$ drawn from $q$:

- Probability of accepting $\hat{x}$: $q(\hat{x}) \cdot \min(1, p(\hat{x})/q(\hat{x})) = \min(q(\hat{x}), p(\hat{x}))$
- If rejected, we sample from the "corrected" distribution: $p'(x) = \text{norm}(\max(0, p(x) - q(x)))$

The total probability of outputting token $x$ (either by acceptance or by sampling from $p'$):
$$P(\text{output} = x) = \min(q(x), p(x)) + P(\text{reject}) \cdot p'(x)$$

$$= \min(q(x), p(x)) + \left(1 - \sum_v \min(q(v), p(v))\right) \cdot \frac{\max(0, p(x) - q(x))}{1 - \sum_v \min(q(v), p(v))}$$

$$= \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$$

(The last equality holds because $\min(a,b) + \max(0, a-b) = a$ for any $a, b \geq 0$.) $\blacksquare$

This exact distributional equivalence is what distinguishes speculative decoding from approximations like top-k filtering or nucleus sampling with draft models.

---

## Intermediate

### Q4. What is the acceptance probability and expected number of accepted tokens per round?

**Answer.**

The **acceptance probability** for a single token (given draft token $\hat{x}_i$ drawn from $q$) is:

$$\alpha_i = \mathbb{E}_{\hat{x} \sim q}\!\left[\min\!\left(1, \frac{p(\hat{x})}{q(\hat{x})}\right)\right] = \sum_x \min(q(x), p(x))$$

This is the overlap between distributions $p$ and $q$: $\alpha = 1 - \text{TV}(p, q)$ where $\text{TV}$ is the total variation distance.

- $\alpha = 1$: $p = q$ (perfect draft model) — all tokens accepted
- $\alpha = 0$: $p$ and $q$ have disjoint support — no tokens accepted

**Expected number of accepted tokens.** Assuming each position has the same acceptance probability $\alpha$ (a simplification; in reality it varies per position):

Tokens 1 through $K$ are accepted independently if we ignore the stopping rule. Actually, with stopping at the first rejection, the number of accepted tokens follows a geometric-like distribution:

$$P(\text{accept exactly } k \text{ of } K) = \alpha^k (1 - \alpha) \text{ for } k < K, \quad P(\text{accept all K}) = \alpha^K$$

**Expected accepted tokens per round** (plus the bonus token, which is always accepted):

$$\mathbb{E}[\text{tokens}] = \sum_{k=0}^{K-1} (k+1)(1-\alpha)\alpha^k + (K+1)\alpha^K$$

$$= \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

This formula is derived by summing the geometric series.

**Speedup.** If the draft model is $\gamma$ times faster than the target model (in wall-clock time per token), the speedup from speculative decoding is approximately:

$$\text{speedup} \approx \frac{\mathbb{E}[\text{tokens}]}{1 + K/\gamma}$$

where the denominator accounts for $K$ draft steps plus 1 target step per round.

See `worked_problems/problem_03_speculative_acceptance_rate.md` for a detailed numerical example.

---

### Q5. What determines draft model quality, and what makes a good draft model choice?

**Answer.**

The key property of a good draft model is **high acceptance rate $\alpha$** — the draft's token distribution $q$ must closely match the target's distribution $p$.

**Factors affecting acceptance rate:**

1. **Draft model size.** Larger draft models produce more target-like distributions. The acceptance rate increases with draft model capacity up to a saturation point (where the draft approaches target quality).

2. **Training data alignment.** The draft model should be trained on the same data distribution as the target. A draft model with different pretraining data will produce systematically different distributions even if it has the same architecture.

3. **Shared tokeniser.** Draft and target must use the same vocabulary; otherwise, token-level comparison is impossible.

4. **Task alignment.** For instruction-following tasks, a draft model trained with similar RLHF/SFT may match the target better than a base model draft.

**Good draft model choices:**

| Target model | Candidate draft models |
|---|---|
| LLaMA-3 70B | LLaMA-3 8B |
| GPT-4 class | GPT-3.5 class |
| Mistral 7B | Mistral 3B or smaller Mistral |
| Claude 3 Opus | Claude 3 Haiku |

**Same family, different size.** The most reliable approach uses a smaller model from the same family and training pipeline. Acceptance rates of $0.7\text{--}0.9$ are typical for well-matched same-family pairs at $K = 4$.

**Self-speculative decoding.** The target model itself can be its own draft model by using early exits — running only the first $m$ of $L$ layers to produce draft tokens, then verifying with the full $L$-layer model. No external draft model is needed, at the cost of needing to store two sets of KV caches.

---

### Q6. When does speculative decoding provide the largest speedup?

**Answer.**

Speculative decoding helps when:

1. **High acceptance rate ($\alpha > 0.7$).** Low acceptance rates mean most draft tokens are rejected, and the overhead of running the draft model outweighs the benefit.

2. **Low batch size.** At batch size 1 (single-user serving), the memory bandwidth bottleneck is most severe. Speculative decoding helps by producing multiple tokens per target model pass. At large batch sizes ($B > 32$), the target model is already compute-bound (matrix multiplications dominate), so the bottleneck is FLOPs, not bandwidth — speculative decoding provides less benefit.

3. **Greedy/near-greedy sampling.** With temperature $T \approx 0$ (near-greedy decoding), the target model's distribution is peaked, so any reasonable draft model tends to produce the highest-probability token — giving high acceptance rates. With high temperature ($T > 1$) or nucleus sampling, the target distribution is more diffuse, and the draft model's predictions are harder to verify.

4. **Repetitive or predictable text.** Code (syntactically constrained), templated responses, and factual recall are easier to predict with a draft model. Open-ended creative generation is harder.

**Speedup range in practice (A100, FP16):**
- Best case (near-greedy, low batch, high acceptance $\alpha = 0.9$): $2.5\text{--}3.5\times$ speedup
- Typical (temperature $\sim 0.7$, batch 1, $\alpha \sim 0.7$): $1.5\text{--}2.5\times$ speedup
- Poor case (high temperature, large batch): $< 1.2\times$ or no benefit

---

## Advanced

### Q7. Derive the optimal lookahead length $K$ that maximises expected throughput.

**Answer.**

Define:
- $t_D$: time for the draft model to generate one token
- $t_T$: time for the target model to perform one forward pass (over any number of tokens, since attention is $O(T^2)$ — but for simplicity, assume $t_T$ is fixed for small $K$)
- $\alpha$: acceptance probability per draft token (assumed equal across positions)

**Time per round** (draft phase + verify phase):
$$\text{time per round} = K \cdot t_D + t_T$$

**Expected tokens per round:**
$$\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

**Expected throughput** (tokens per second):
$$\text{throughput}(K) = \frac{\mathbb{E}[\text{tokens}]}{\text{time per round}} = \frac{(1 - \alpha^{K+1})/(1 - \alpha)}{K t_D + t_T}$$

**Maximise over $K$.** Taking the derivative and setting to zero:

$$\frac{d}{dK}\left[\frac{1 - \alpha^{K+1}}{(1-\alpha)(Kt_D + t_T)}\right] = 0$$

Using the quotient rule:
$$-\alpha^{K+1} \ln\alpha \cdot (Kt_D + t_T) - (1 - \alpha^{K+1}) t_D = 0$$

$$-\alpha^{K+1} \ln\alpha \cdot (Kt_D + t_T) = (1 - \alpha^{K+1}) t_D$$

This has no closed form in general. In the high-acceptance limit ($\alpha \to 1$): $1 - \alpha^{K+1} \approx (K+1)(1-\alpha)$, and the throughput becomes:

$$\frac{(K+1)(1-\alpha)/(1-\alpha)}{Kt_D + t_T} = \frac{K+1}{Kt_D + t_T}$$

Maximising $\frac{K+1}{Kt_D + t_T}$ over $K$: for large $K$, the ratio $\to 1/t_D$ — throughput is limited by the draft model speed. The optimal $K$ is large when $\alpha$ is high and the draft is fast.

**Practical rule of thumb.** For a draft model with $t_D = t_T / 10$ (10x faster) and acceptance $\alpha = 0.7$:

$$K^* \approx \frac{-\log(1/t_D \cdot t_T)}{\log(1/\alpha)} = \frac{\log 10}{\log(1/0.7)} = \frac{2.303}{0.357} \approx 6$$

Optimal $K \approx 6$. Beyond $K = 6$, the decreasing marginal acceptance rate makes each additional draft step less valuable than the latency it adds.

---

### Q8. What is SpecTr and how does tree-based speculative decoding improve on vanilla speculative decoding?

**Answer.**

**Limitation of vanilla speculative decoding.** The draft model generates a single sequence of $K$ tokens. If the first token is rejected, all subsequent draft tokens are discarded — work is wasted.

**Tree-based speculation (SpecTr, Speculative Tree Decoding).** Instead of a single draft sequence, generate a tree of possible continuations:

- From the current token, draft $b_1$ candidate tokens (branching factor)
- For each of those, draft $b_2$ candidates
- Continue for depth $d$: total draft sequences $= b_1 \times b_2 \times \ldots \times b_d$

**Verification.** The target model verifies all draft sequences simultaneously in one forward pass, using a tree attention mask (each token attends to its ancestors but not siblings).

**Accept the longest valid path.** Walk the tree greedily: for each level, accept the highest-probability accepted token among siblings and continue from there.

**Expected tokens accepted.** With a balanced tree of branching factor $b$ and depth $d$:
$$\mathbb{E}[\text{tokens}] \approx \frac{1 - (b\alpha)^{d+1}}{1 - b\alpha} \quad \text{if } b\alpha < 1$$

For $b = 3$, $d = 3$, $\alpha = 0.7$: $b\alpha = 2.1 > 1$, so the tree formulation saturates — the effective acceptance rate of the best branch in a tree of 3 candidates is much higher than a single sequence.

**Practical gains.** SpecTr and related methods (Medusa, which uses multiple heads on the draft model to generate tree branches) achieve $2\text{--}3\times$ additional speedup over vanilla speculative decoding — potentially $5\text{--}8\times$ over standard autoregressive decoding for suitable tasks.

**Trade-off.** Tree-based verification requires a tree attention mask, which is more complex to implement and less cache-friendly. The benefit is a much higher effective acceptance rate at the same compute budget.
