# Quantisation: PTQ and QLoRA

## Overview

Quantisation reduces the numerical precision of model weights (and optionally activations), enabling larger models to fit in limited GPU memory and accelerating inference by reducing memory bandwidth requirements. This file covers post-training quantisation methods and QLoRA for fine-tuning quantised models.

---

## Fundamentals

### Q1. What is quantisation and why does it reduce memory and improve speed?

**Answer.**

Quantisation maps floating-point values to a lower-precision representation. For weights, this means storing each parameter in fewer bits:

| Format | Bits | Range | Memory per parameter |
|---|---|---|---|
| FP32 | 32 | $\pm 3.4 \times 10^{38}$ | 4 bytes |
| BF16/FP16 | 16 | BF16: $\pm 3.4 \times 10^{38}$; FP16: $\pm 65504$ | 2 bytes |
| INT8 | 8 | $-128$ to $127$ (signed) | 1 byte |
| INT4/NF4 | 4 | 16 distinct values | 0.5 bytes |

**Memory savings.** An LLM with $P$ parameters stored in INT8 uses $P$ bytes vs $2P$ bytes in FP16 — a $2\times$ reduction. In INT4, it is $P/2$ bytes — a $4\times$ reduction.

**Speed improvement (memory bandwidth).** Decode-phase inference is memory-bandwidth-limited (see `kv_cache_mechanism.md`). If the bottleneck is reading weights from HBM, halving weight size halves HBM reads, doubling throughput. This is the primary speed benefit of weight quantisation (W4A16, W8A16 schemes).

**Compute improvement (W8A8).** If both weights and activations are quantised to INT8, matrix multiplications can run on INT8 tensor cores (available on A100, H100) which have $2\times$ higher throughput than FP16 tensor cores. This improves compute-bound scenarios (large batch inference, prefill).

---

### Q2. What is the basic uniform quantisation formula?

**Answer.**

**Affine (asymmetric) quantisation.** Maps float $x$ to integer $x_q$:

$$x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x - z}{s}\right),\; q_{\min},\; q_{\max}\right)$$

Dequantisation:
$$\hat{x} = s \cdot x_q + z$$

where:
- $s > 0$ is the **scale** (step size between quantisation levels)
- $z$ is the **zero point** (the float value that maps to integer 0)
- $q_{\min}, q_{\max}$ are the integer range bounds (e.g., $-128, 127$ for INT8)

**Symmetric quantisation.** Forces $z = 0$, simplifying to:
$$x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right),\; -q_{\max},\; q_{\max}\right)$$
$$s = \frac{\max(|x|)}{q_{\max}}$$

Symmetric is computationally simpler (no zero-point arithmetic) and is used for weight quantisation in most LLM quantisation schemes.

**Scale selection.** For a tensor with values in $[\alpha, \beta]$:
$$s = \frac{\beta - \alpha}{2^b - 1}, \quad z = -\text{round}(\alpha / s)$$

where $b$ is the number of bits.

---

### Q3. What is Post-Training Quantisation (PTQ) and what are its variants?

**Answer.**

PTQ quantises a trained model without any further training or backpropagation. The model is quantised once and deployed. This contrasts with Quantisation-Aware Training (QAT), which simulates quantisation during training.

**PTQ variants by what is quantised:**

| Method | Weights | Activations | Notes |
|---|---|---|---|
| W8A16 | INT8 | FP16 | Most common; good quality/speed |
| W4A16 | INT4 | FP16 | $2\times$ smaller than W8A16; slight quality loss |
| W8A8 | INT8 | INT8 | Requires INT8 activations; high throughput |
| W4A8 | INT4 | INT8 | Aggressive; needs calibration |

**W4A16** is currently the dominant production format for LLM serving: weights stored in INT4 reduce memory bandwidth $4\times$ vs FP32, while keeping activations in FP16 avoids activation quantisation error (which is harder to control than weight quantisation error).

---

## Intermediate

### Q4. Explain GPTQ. What makes it more accurate than naive round-to-nearest quantisation?

**Answer.**

GPTQ (Frantar et al., 2022) is a layer-wise, data-driven post-training quantisation method that minimises the weight quantisation error using second-order information.

**Problem.** For a linear layer with weight matrix $W$ and inputs $X$, the output error from quantising $W$ to $\hat{W}$ is:
$$\mathcal{E} = \|WX - \hat{W}X\|_F^2 = \|(W - \hat{W})X\|_F^2$$

Minimising this with respect to the quantised values is more accurate than minimising $\|W - \hat{W}\|_F^2$ (which ignores the input distribution).

**OBS framework.** GPTQ builds on the Optimal Brain Surgeon framework:
1. Quantise one weight $w_{ij}$ at a time
2. After quantising $w_{ij}$, compute the optimal compensation adjustment to all remaining weights using the inverse Hessian:

$$\delta w_{-ij} = -\frac{w_{ij} - \hat{w}_{ij}}{[H^{-1}]_{jj}} H^{-1}[:, j]$$

where $H = 2 X X^T$ is the Hessian of the layer-wise error with respect to weights.

**Practical speedups in GPTQ:**
- Columns are quantised in order (left to right)
- The Hessian inverse is updated lazily (Cholesky decomposition)
- Quantisation runs column-by-column on GPU in $\sim 1\text{--}4$ hours for a 70B model

**Accuracy.** GPTQ INT4 typically achieves perplexity within $0.1\text{--}0.5$ bits/token of FP16 on standard benchmarks — far better than naive round-to-nearest at the same bit width.

---

### Q5. What is AWQ (Activation-aware Weight Quantisation)?

**Answer.**

AWQ (Lin et al., 2023) observes that not all weights are equally important: **salient weights** corresponding to large-magnitude activation channels cause disproportionate quantisation error if quantised to low precision.

**Key insight.** For a linear layer $y = Wx$, if input channel $i$ has large activations $|x_i| \gg$ average, then even small quantisation error in $W[:,i]$ produces large output error. A small fraction ($\sim 1\%$) of weight channels are salient, determined by input activation statistics.

**AWQ approach:**
1. Collect per-channel activation statistics on a small calibration dataset
2. Find a per-channel scale $s_i > 1$ for salient channels such that quantisation error is minimised
3. Scale the weights: $W' = W \cdot \text{diag}(s)^{-1}$, compensated by scaling the input: $x' = x \cdot \text{diag}(s)$
4. Quantise $W'$ (the rescaled weights)

**Effect.** By scaling salient weight columns by $1/s_i < 1$ before quantisation, the range of those columns is compressed, reducing the quantisation step size and thus the error for those critical channels.

**Comparison to GPTQ:**
- AWQ is faster to run (minutes vs hours for large models)
- AWQ does not modify weights within a column (only per-column scaling), preserving more structure
- Both achieve similar accuracy on standard benchmarks; AWQ sometimes better on reasoning tasks

---

### Q6. Explain LoRA. What is the mathematical basis for why it works?

**Answer.**

LoRA (Low-Rank Adaptation; Hu et al., 2022) fine-tunes pretrained models by adding low-rank weight updates:

$$W' = W_0 + \Delta W = W_0 + BA$$

where $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is frozen, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, $A \in \mathbb{R}^{r \times d_{\text{in}}}$, and $r \ll \min(d_{\text{out}}, d_{\text{in}})$.

**Mathematical basis.** The hypothesis is that the weight changes $\Delta W$ during fine-tuning have low intrinsic rank. Empirically, the rank of effective weight updates in fine-tuning is much smaller than the full rank of the weight matrix. For instance, for a matrix of rank $d = 4096$, the effective rank of fine-tuning updates is often $r \leq 64$.

**Parameter reduction.** The adapter $BA$ requires:
$$r \times (d_{\text{in}} + d_{\text{out}}) \text{ parameters vs } d_{\text{in}} \times d_{\text{out}} \text{ for the full matrix}$$

For $d_{\text{in}} = d_{\text{out}} = 4096$, $r = 8$:
$$\frac{8 \times 8192}{4096^2} = \frac{65536}{16777216} \approx 0.4\%$$

LoRA adapts $< 1\%$ of parameters while achieving fine-tuning quality close to full fine-tuning.

**Scaling at inference.** The matrices $A$ and $B$ are multiplied $\Delta W = BA$ and merged with $W_0$ before serving: $W' = W_0 + BA$. This adds zero inference overhead.

---

### Q7. What is QLoRA? How does it combine quantisation and LoRA?

**Answer.**

QLoRA (Dettmers et al., 2023) enables fine-tuning of very large quantised models on consumer or single-GPU hardware by combining:

1. **NF4 quantisation** of the base model weights (4-bit)
2. **Double quantisation** of the quantisation constants
3. **LoRA adapters** (full precision) for the trainable parameters
4. **Paged optimisers** to handle memory spikes

**QLoRA training procedure:**

1. Load the pretrained base model in NF4 format (4-bit)
2. Freeze all base model weights (no gradient through quantised weights)
3. Add LoRA adapters in BF16 to the attention projections
4. During forward pass:
   - Dequantise base model weights on-the-fly: NF4 $\to$ BF16
   - Compute the layer output using the dequantised weights plus the LoRA contribution
5. Backpropagate only through the LoRA adapters
6. Update only the LoRA parameters

**Memory analysis.** For LLaMA-2 65B:
- FP16 weights: $65\text{B} \times 2 \approx 130$ GB — requires multiple A100s
- QLoRA NF4 weights: $65\text{B} \times 0.5 \approx 32$ GB — fits on a single A100 80GB
- LoRA adapter overhead: $\sim 0.1\text{--}0.5\%$ of NF4 weight memory

**Quality.** QLoRA achieves fine-tuning quality close to full 16-bit fine-tuning, enabling instruction-tuning of 65B models on a single GPU — a $10\times$ reduction in hardware cost.

---

### Q8. What is the NF4 data type and why is it designed for normally distributed weights?

**Answer.**

NF4 (Normal Float 4-bit; Dettmers et al., 2023) is a data type specifically designed for quantising neural network weights, which are approximately normally distributed $\mathcal{N}(0, \sigma^2)$ after normalisation.

**Standard INT4 quantisation.** Places quantisation levels uniformly across $[-\text{max}, \text{max}]$. For a normal distribution, this is suboptimal: most weight values cluster near 0, so many quantisation levels near 0 are wasted in representing the tails.

**NF4 design.** Places quantisation levels at the quantiles of the standard normal distribution. Specifically:

The 16 NF4 levels are the values $q_i$ such that:
$$q_i = \Phi^{-1}\!\left(\frac{i}{15}\right) \text{ normalised to } [-1, 1]$$

where $\Phi^{-1}$ is the inverse normal CDF.

This places more quantisation levels near the mean (where most weights concentrate) and fewer in the tails — minimising the expected quantisation error for normally distributed weights.

**Result.** NF4 is information-theoretically optimal for normally-distributed data: it minimises the expected $\ell_2$ quantisation error for a $\mathcal{N}(0, 1)$ distribution at 4-bit precision.

**Double quantisation.** The NF4 quantisation uses a per-block scale constant (one float per 64 weights). For a large model, these constants themselves consume non-trivial memory. QLoRA quantises these constants a second time (to 8-bit), reducing their overhead from 0.5 bits/weight ($32\text{ bits} / 64$) to $0.127$ bits/weight — saving $\sim 0.37$ bits/weight, or $\sim 3.6$ GB for a 65B model.

---

## Advanced

### Q9. Why is activation quantisation (W8A8) harder to achieve than weight quantisation (W8A16)?

**Answer.**

**Weights are static; activations are dynamic.** Weights are fixed after training, so their distribution can be measured once (using a calibration dataset) and a fixed scale can be set. Activations change with every input, making it harder to choose a scale that works well universally.

**Outlier activations.** In large transformer models, activation tensors often contain a small number of extreme outlier values — magnitudes $10\text{--}100\times$ the typical value. These outliers appear systematically in specific channels across different inputs (LLM.int8 analysis, Dettmers et al., 2022).

If a naive per-tensor scale is used, the scale is set by the outlier value, leaving most quantisation levels unused for the normal-range values — catastrophic precision loss.

**SmoothQuant** (Xiao et al., 2023) addresses this by migrating the quantisation difficulty from activations to weights. For each channel $i$:

$$y = (x \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W^T)$$

where $s_i = \max(|x_i|)^\alpha / \max(|w_i|)^{1-\alpha}$ balances the scale between inputs and weights. After this channel-wise rescaling, both $x \cdot \text{diag}(s)^{-1}$ and $\text{diag}(s) \cdot W$ have more uniform magnitudes, enabling effective INT8 quantisation for both.

**LLM.int8()** takes a different approach: keep outlier channels in FP16 and quantise the remaining channels in INT8. The matrix multiplication is split: outlier channels use FP16; non-outlier channels use INT8. This recovers near-zero degradation with a small compute overhead from the FP16 outlier path.
