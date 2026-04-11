# Training Stability for Transformers

## Overview

Training large Transformer models is notoriously fragile. Loss spikes, divergence,
and slow convergence are common failure modes. This file covers the techniques
practitioners use to keep training stable, how to diagnose problems when they
occur, and the reasoning behind each intervention.

---

## Tier 1 — Fundamentals

### Q1. What is gradient clipping and why is it necessary for Transformer training?

**Answer.**

Gradient clipping bounds the norm of the gradient vector before the optimiser
update step. The most common variant is **global norm clipping**:

$$
g \leftarrow g \cdot \frac{\min(\tau, \|g\|_2)}{\|g\|_2}
$$

where $g$ is the concatenated gradient vector across all parameters and $\tau$
is the clip threshold (typically $1.0$ for language models).

**Why it is necessary.** Transformers trained with Adam can experience sudden
gradient spikes — often $10\times$ to $1000\times$ the normal magnitude — that
push weights far outside the region of good loss landscape curvature. Without
clipping, a single bad mini-batch can permanently damage a run. Gradient clipping
converts these spikes into a bounded step in the gradient direction, preserving
the direction of the update while limiting the magnitude.

**Common mistake.** Clipping per-parameter (rather than globally) can distort the
relative scaling between layers, which is usually undesirable. Global norm
clipping is preferred.

**Practical note.** Monitor the gradient norm as a training metric. A norm that
is consistently at the clip threshold means the threshold is too low. A norm that
spikes to the threshold occasionally is expected and healthy. A norm that is
chronically near zero can indicate vanishing gradients.

---

### Q2. What is Xavier (Glorot) initialisation, and when should you use it?

**Answer.**

Xavier initialisation sets each weight $W \in \mathbb{R}^{n_{in} \times n_{out}}$
by drawing from:

$$
W \sim \mathcal{U}\!\left[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}},\; \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\right]
\quad \text{(uniform variant)}
$$

or equivalently from $\mathcal{N}(0,\, \sigma^2)$ where
$\sigma = \sqrt{\tfrac{2}{n_{in}+n_{out}}}$.

**The derivation goal.** Xavier initialisation is derived under the assumption of
linear (or approximately linear) activations. It seeks to preserve the variance
of activations and gradients across layers, so that neither signal explodes nor
vanishes as it propagates through depth.

**When to use it.** Appropriate for layers followed by $\tanh$ or sigmoid
activations, and widely used as a default for Transformer attention projections
and feed-forward layers where outputs are further normalised.

**He initialisation** (for ReLU activations) uses $\sigma = \sqrt{\tfrac{2}{n_{in}}}$,
doubling the variance to compensate for ReLU zeroing approximately half of its
inputs. Use He for ReLU/GELU feed-forward layers; use Xavier when activations are
more symmetric.

---

### Q3. What is the difference between FP16 and BF16 for training, and why does the choice matter?

**Answer.**

Both formats occupy 16 bits but allocate those bits differently:

| Format | Sign | Exponent | Mantissa | Max value | Min normal |
|--------|------|----------|----------|-----------|------------|
| FP32   | 1    | 8        | 23       | ~3.4e38   | ~1.2e-38   |
| FP16   | 1    | 5        | 10       | ~6.5e4    | ~6.1e-5    |
| BF16   | 1    | 8        | 7        | ~3.4e38   | ~1.2e-38   |

**Key difference.** BF16 has the same exponent range as FP32, so overflow/underflow
is not a concern during forward or backward passes. FP16's narrow exponent range
means activations or gradients that exceed ~65,000 will overflow to infinity,
immediately destabilising training.

**Practical implication.** FP16 training requires **loss scaling** (multiplying the
loss by a large scalar before backward, then dividing the gradients back) to keep
gradient values inside the representable range. BF16 does not require loss scaling,
making it substantially simpler to implement and more robust in practice. Most
modern training frameworks (PyTorch AMP with `dtype=torch.bfloat16`, JAX, etc.)
default to BF16 when the hardware supports it (A100, H100, TPU v3+).

---

### Q4. What is gradient accumulation and what problem does it solve?

**Answer.**

Gradient accumulation runs $k$ forward-backward passes on small mini-batches
before calling the optimiser update, summing (or averaging) gradients across
those $k$ steps. The effective batch size seen by the optimiser becomes
$k \times B_{\text{micro}}$.

**The problem it solves.** Large batch training is beneficial for optimiser
efficiency and throughput, but the physical batch size is bounded by GPU memory.
Gradient accumulation decouples the **effective batch size** from the **memory
footprint**, allowing a single GPU with limited VRAM to match the optimiser
behaviour of a multi-GPU setup.

**Important caveat.** When using batch normalisation (rare in Transformers),
statistics computed on the micro-batch are not equivalent to those computed on
the full effective batch. Layer normalisation, which Transformers use, has no
such issue because it normalises over the token/feature dimension, not the batch
dimension.

---

## Tier 2 — Intermediate

### Q5. A training run has been stable for 50,000 steps and then experiences a sudden loss spike. Walk through your diagnostic process.

**Answer.**

A loss spike after stable training is one of the most common failure modes in
large model runs. The diagnostic process proceeds in order of ease and likelihood:

**Step 1 — Check the gradient norm log.**
A spike in gradient norm immediately preceding the loss spike strongly implicates
a bad mini-batch or a data corruption event. If gradient clipping was active,
check whether the norm was hitting the clip threshold regularly in the thousands
of steps before the spike — this can indicate the model was slowly accumulating
instability.

**Step 2 — Inspect the data pipeline.**
Bad batches are a surprisingly common cause: corrupted documents, runaway-length
sequences that exceed the model's positional encoding range, or tokenisation
artefacts. Replay the data loader state at step $N-5$ to $N$ and examine the
actual inputs.

**Step 3 — Check for NaN/Inf propagation.**
Enable `detect_anomaly` (PyTorch) or equivalent. Loss scaling overflow in FP16
training produces NaN gradients; the loss scaler should have reduced the scale
factor automatically, but a misconfigured scaler will not. Check the loss scaling
history.

**Step 4 — Examine the learning rate schedule.**
Warmup schedules that end abruptly, or cosine schedules that restart, can
occasionally produce a transient spike. Correlate the spike with the LR schedule.

**Step 5 — Check for hardware faults.**
Bit-flip errors in GPU HBM are rare but real. If the spike is accompanied by
suspiciously large weight updates in a specific layer, consider re-running that
checkpoint on different hardware.

**Recovery strategies.** Roll back to the last healthy checkpoint (typically 1,000
steps before the spike). Some practitioners add "loss spike rollback" to their
training loop: if $\mathcal{L}_{t} > \alpha \cdot \mathcal{L}_{t-1}$ (where
$\alpha \approx 1.5\text{–}3$), automatically revert and skip the offending batch.

---

### Q6. Explain mixed-precision training with FP32 master weights. Why keep a full-precision copy of the weights?

**Answer.**

Mixed-precision training (Micikevicius et al., 2018) maintains two copies of the
model parameters:

1. **FP16/BF16 working copy** — used for forward pass and backward pass. The
   lower precision reduces memory bandwidth consumption and enables tensor core
   acceleration (2–4$\times$ speedup on A100).

2. **FP32 master copy** — used only for the optimiser update.

**Why the FP32 master copy is necessary.** The optimiser update adds a small
delta to the weights:

$$
\theta \leftarrow \theta - \eta \cdot \hat{g}
$$

When $\eta$ is small (e.g., $10^{-4}$) and $\|\hat{g}\|$ is of order 1, the
update $\eta \hat{g}$ can be as small as $10^{-7}$. FP16 has a minimum normal
value of ~$6 \times 10^{-5}$ and a mantissa precision of ~$10^{-3}$. An update
of $10^{-7}$ is simply rounded to zero in FP16 — the weights would never change.
FP32's 23-bit mantissa (precision ~$10^{-7}$) preserves these small updates.

**Memory cost.** Maintaining an FP32 master copy doubles the weight memory
relative to an all-FP16 training scheme, but halves it relative to all-FP32.
With Adam, the optimiser states (first and second moment) are also stored in
FP32 and constitute the dominant memory cost.

**Memory breakdown for a model with $N$ parameters:**

| Item | Precision | Bytes per param |
|------|-----------|-----------------|
| Working weights | BF16 | 2 |
| Master weights | FP32 | 4 |
| Adam $m$ (first moment) | FP32 | 4 |
| Adam $v$ (second moment) | FP32 | 4 |
| **Total** | | **14** |

For a 7B parameter model this is approximately 98 GB — a typical A100-80GB
cluster requires at least two GPUs for the parameters and optimiser states alone,
before activations.

---

### Q7. Compare data parallelism and model parallelism. When does each become the bottleneck?

**Answer.**

**Data parallelism (DP).** Each GPU holds a full copy of the model. Each GPU
processes a different shard of the mini-batch. Gradients are all-reduced across
GPUs after the backward pass. The optimiser step runs independently on each GPU
(with DDP) or on a single GPU (with parameter server).

- Scales throughput linearly with GPU count (up to communication overhead).
- Bottleneck: the full model must fit on a single GPU. For 7B+ parameter models,
  this requires activation checkpointing or ZeRO optimiser sharding (which is a
  form of DP + partial model parallelism).

**Tensor parallelism (TP, a form of model parallelism).** Individual weight
matrices are sharded across GPUs. For a linear layer $Y = XW$, the columns of
$W$ are distributed so each GPU computes a partial result; an all-reduce combines
them. Megatron-LM uses this within the attention and FFN layers.

- Requires high-bandwidth intra-node interconnects (NVLink). Cross-node TP is
  typically impractical.
- Bottleneck: all-reduce latency within each forward/backward pass; does not
  scale well beyond 8 GPUs per model shard.

**Pipeline parallelism (PP).** Layers are partitioned across GPUs; micro-batches
flow through the pipeline. GPUs 0...$k-1$ hold the first $k$ layers, etc.

- Enables training models that span many nodes.
- Bottleneck: pipeline bubbles (idle GPU time while waiting for the previous
  stage). Mitigated by micro-batching (GPipe) or interleaved schedules
  (Megatron-LM).

**In practice**, large model training combines all three (3D parallelism): DP
across replicas, TP within a node, PP across nodes.

---

### Q8. What is the Pre-LN vs Post-LN architecture choice and how does it affect training stability?

**Answer.**

In the **Post-LN** (original Transformer) arrangement, layer normalisation is
applied after the residual addition:

$$
x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))
$$

In **Pre-LN**, it is applied before the sublayer, inside the residual branch:

$$
x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))
$$

**Stability implications.** In Post-LN, the gradients flowing back through deep
networks must pass through many LayerNorm operations whose scale parameters can
amplify or suppress the signal. This tends to produce very large gradients at
the early layers, requiring careful warmup schedules and often causing divergence
without them (Xiong et al., 2020 showed that Post-LN requires warmup; Pre-LN
does not strictly require it).

Pre-LN guarantees that a direct gradient path exists from the loss to the
earliest layer without passing through any sublayer (via the residual stream),
making the effective depth for gradient flow much shallower and training more
robust.

**Trade-off.** Pre-LN converges to slightly worse final perplexity than Post-LN
for the same number of steps (the residual stream is not normalised at the
output, leading to representation collapse concerns at depth). The "Sandwich
normalisation" and "DeepNorm" (Su et al., 2022) variants attempt to recover
Post-LN quality while preserving Pre-LN stability.

---

## Tier 3 — Advanced

### Q9. Derive why the standard Transformer initialisation at depth $d$ produces $O(d)$ variance growth and explain how $1/\sqrt{d}$ rescaling of residual branches fixes it.

**Answer.**

**Setup.** Consider a Transformer with $L$ layers. Each layer adds a residual
contribution: $x_{l+1} = x_l + F_l(x_l)$. Assume that $F_l(x_l)$ has zero
mean and variance $\sigma_F^2$ (independent of $x_l$ for this analysis), and
that residuals are approximately uncorrelated across layers.

**Variance accumulation.** Treating layers as independent:

$$
\text{Var}(x_L) = \text{Var}(x_0) + L \cdot \sigma_F^2
$$

The variance grows linearly with depth $L$. For deep models ($L = 96$ in
GPT-3), $\text{Var}(x_L)$ can be $\sim 100\times$ larger than at the input,
causing activation overflow and unstable softmax in attention.

**The fix.** Scale each residual branch output by $\alpha = 1/\sqrt{2L}$ (or
a similar $O(1/\sqrt{L})$ factor). Then:

$$
\text{Var}(x_L) = \text{Var}(x_0) + L \cdot \alpha^2 \sigma_F^2
= \text{Var}(x_0) + L \cdot \frac{\sigma_F^2}{2L}
= \text{Var}(x_0) + \frac{\sigma_F^2}{2}
$$

The variance is now $O(1)$ regardless of depth.

**Implementations.** GPT-2 used a $1/\sqrt{N_l}$ scaling of the weights of the
second linear layer in each residual block at initialisation, where $N_l$ is the
number of residual layers. DeepNet (Su et al., 2022) formalises this with a
constant $\alpha$ multiplier and $\beta$ normalisation init, proving convergence
for 1,000-layer Transformers.

---

### Q10. Explain the role of loss scaling in FP16 training. How does dynamic loss scaling work and what are its failure modes?

**Answer.**

**The problem.** Gradients in backward pass occupy a range roughly $[10^{-7}, 10^0]$.
FP16 cannot represent values smaller than the minimum subnormal (~$6 \times 10^{-8}$)
and rounds values in $[6 \times 10^{-8}, 6 \times 10^{-5}]$ to subnormals (which
are imprecise) or zero. A large fraction of gradient values therefore flush to
zero, producing biased gradient estimates.

**Static loss scaling.** Multiply the scalar loss by a constant $s$ (e.g., $2^{15}$)
before calling `.backward()`. All gradients are scaled by $s$, shifting them into
the representable range. Before the optimiser step, divide gradients by $s$:

$$
\tilde{\mathcal{L}} = s \cdot \mathcal{L}
\implies \nabla_\theta \tilde{\mathcal{L}} = s \cdot \nabla_\theta \mathcal{L}
$$

If gradients are in $[10^{-7}, 10^0]$, scaling by $2^{15} \approx 32768$ maps
them to $[3 \times 10^{-3}, 3 \times 10^4]$, well within FP16 range.

**Dynamic loss scaling.** The ideal scale changes during training. Dynamic
scaling starts with a large scale (e.g., $2^{24}$) and:

1. After each successful step, tentatively increase the scale by a factor (e.g.,
   multiply by 2 every 2,000 steps).
2. If any gradient is NaN or Inf (indicating overflow), skip the optimiser step,
   reduce the scale by half, and continue.

**Failure modes.**

- **Underflow despite scaling.** If the model produces extremely small gradients
  (e.g., at initialisation with very small LR), even a large scale may not lift
  them above zero. Usually transient.
- **Infinite scale oscillation.** If the model is near divergence, gradients
  overflow immediately after every scale increase, locking the scaler into
  constant halving. Monitor `loss_scale` in your training logs; if it is
  collapsing toward 1, the run is likely diverging for other reasons.
- **Stale skipped steps.** Skipped steps due to overflow do not advance the LR
  schedule or the data iterator in naive implementations, causing subtle train/val
  splits to drift if many steps are skipped. PyTorch's GradScaler handles this
  correctly; custom implementations may not.

**Why BF16 avoids this entirely.** BF16's 8-bit exponent matches FP32, so no
gradient value that is representable in FP32 will overflow or underflow in BF16.
Loss scaling is unnecessary.

---

### Q11. What is the "critical batch size" and how does it interact with training stability and efficiency?

**Answer.**

The **critical batch size** $B_\text{crit}$ is the batch size at which the
gradient noise scale $\mathcal{N} = \text{tr}(\Sigma) / \|\bar{g}\|^2$ equals 1,
where $\Sigma$ is the gradient covariance and $\bar{g}$ is the mean gradient. It
separates two regimes:

- $B \ll B_\text{crit}$: **noise-limited regime.** Each step provides a noisy
  gradient estimate. More steps are beneficial; increasing $B$ reduces noise
  approximately linearly and each step is proportionally more useful.
- $B \gg B_\text{crit}$: **compute-limited regime.** Gradient estimates are
  already near-deterministic. Doubling $B$ barely improves per-step quality but
  doubles compute cost. Throughput efficiency drops.

**Measurement.** $B_\text{crit}$ can be estimated empirically by training with
several batch sizes and fitting the relationship:

$$
\frac{1}{\varepsilon_\text{steps}} \propto \frac{1}{B} + \frac{1}{B_\text{crit}}
$$

where $\varepsilon_\text{steps}$ is a measure of how much each step reduces loss.

**Stability implication.** Very large batches (much larger than $B_\text{crit}$)
require linear scaling of the learning rate (the linear scaling rule: Goyal et
al., 2017) to maintain equivalent loss trajectories. But large LRs increase the
risk of instability. This is why simply "use the biggest batch that fits on GPUs"
is not always the right strategy — throughput and optimiser quality are in tension.

**Typical values.** For large language models, $B_\text{crit}$ tends to be on the
order of $10^5\text{–}10^6$ tokens. GPT-3 was trained with a batch size of $\sim 3.2M$
tokens, well above $B_\text{crit}$, prioritising throughput over per-step
efficiency.

---

### Q12. A colleague proposes training a 70B model entirely in INT8 to save memory. What are the stability risks and how would you address them?

**Answer.**

**Risks of INT8 forward and backward pass.**

1. **Gradient quantisation error.** INT8 has 256 discrete levels. Gradients span
   many orders of magnitude during training; quantising them to INT8 introduces
   relative errors of up to $1/128 \approx 0.78\%$ per element, but outlier
   channels (which are common in Transformer activations, as shown by Dettmers
   et al., 2022) introduce errors several orders of magnitude larger.

2. **Outlier features.** Specific dimensions in Transformer hidden states
   regularly take values $50\times$ to $100\times$ larger than the median. INT8
   per-tensor quantisation must accommodate these outliers by using a large scale
   factor, which compresses all other values into only a few quantisation levels.

3. **Gradient accumulation precision loss.** Accumulating INT8 gradients requires
   periodic upcasting; if done naively, numerical errors compound.

**Practical approach for inference-only INT8.** LLM.int8() (Dettmers et al., 2022)
isolates outlier dimensions and performs those as FP16 matrix multiplications while
quantising the rest to INT8. This preserves accuracy and is only used for inference.

**Practical approach for training.** Full INT8 training is an active research area
(e.g., FP8 training on H100 is more tractable). The recommended approach for
memory-efficient training is:

- Use BF16 for forward/backward (activations and weights).
- Use INT8/NF4 weight quantisation with QLoRA-style adapters if the goal is
  fine-tuning with limited memory, not full pretraining.
- For full pretraining: ZeRO-3 optimiser sharding (DeepSpeed) is more
  appropriate than INT8 training to reduce memory footprint.

**Summary.** INT8 training for a 70B model is not a well-established practice
as of 2025. The risks of instability from gradient quantisation outweigh the
memory savings. FP8 training (H100+) or ZeRO-3 BF16 are the appropriate tools.
