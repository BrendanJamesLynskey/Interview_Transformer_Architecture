# RMSNorm and Pre-Normalisation

## Overview

Normalisation placement and formulation are small architectural choices with large downstream effects on training stability, learning rate sensitivity, and model quality. This file covers the LayerNorm vs RMSNorm distinction, pre-norm vs post-norm placement, and the emerging QK-norm technique.

---

## Fundamentals

### Q1. What is Layer Normalisation and what problem does it solve?

**Answer.**

Layer Normalisation (Ba et al., 2016) normalises each token's representation across the feature dimension, making the distribution of activations stable across training:

$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

where:
- $\mu = \frac{1}{d} \sum_{i=1}^d x_i$ is the feature-wise mean
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$ is the feature-wise variance
- $\gamma, \beta \in \mathbb{R}^d$ are learnable scale and shift parameters
- $\epsilon$ is a small constant for numerical stability (typically $10^{-5}$)

**What it solves.** Deep networks suffer from "internal covariate shift": the distribution of activations at each layer changes during training as the parameters of earlier layers change. This requires very careful learning rate tuning and causes slow convergence. LayerNorm stabilises the distribution of activations, enabling larger learning rates and more robust training.

**Why LayerNorm (not BatchNorm) in transformers.** BatchNorm normalises across the batch dimension, requiring large batches for stable statistics and behaving differently at train vs test time. Transformers process variable-length sequences; LayerNorm normalises within each token independently, requiring no batch statistics. This makes it compatible with any sequence length and batch size.

---

### Q2. What is RMSNorm and how does it differ from LayerNorm?

**Answer.**

RMSNorm (Zhang & Sennrich, 2019) simplifies LayerNorm by removing the mean-centring step:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \gamma \quad \text{where } \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}$$

Note: no mean subtraction, no bias parameter $\beta$.

**Difference from LayerNorm:**

| Property | LayerNorm | RMSNorm |
|---|---|---|
| Mean subtraction | Yes: $\mathbf{x} \leftarrow \mathbf{x} - \mu$ | No |
| Normalisation divisor | $\sqrt{\sigma^2 + \epsilon}$ (variance of centred $x$) | $\sqrt{\text{mean}(x^2) + \epsilon}$ (RMS of $x$) |
| Learnable parameters | $\gamma$ (scale) and $\beta$ (shift) | $\gamma$ (scale) only |
| Parameter count | $2d$ | $d$ |
| Invariance | Shift-invariant and scale-invariant | Scale-invariant only |

**Computational advantage.** RMSNorm requires computing one statistic (RMS) instead of two (mean and variance), and skips the subtraction. On modern hardware, this is approximately $7\text{--}15\%$ faster per normalisation layer.

**Hypothesis.** The re-centring step in LayerNorm may be unnecessary for training stability in large models. The scale normalisation (dividing by RMS or std) is what prevents activation explosion; the mean centring adds invariance to additive bias but is less critical. This hypothesis is supported empirically: LLaMA, Mistral, Gemma, and PaLM all use RMSNorm with no quality degradation relative to LayerNorm.

---

### Q3. What is the difference between pre-norm and post-norm placement?

**Answer.**

The normalisation layer can be placed in two positions relative to the sub-layers (attention or FFN):

**Post-norm (original Transformer):**
$$\mathbf{x}' = \text{LN}(\mathbf{x} + \text{Attention}(\mathbf{x}))$$
$$\mathbf{x}'' = \text{LN}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$

Normalisation is applied after the residual connection, to the sum of input and sub-layer output.

**Pre-norm (modern LLMs):**
$$\mathbf{x}' = \mathbf{x} + \text{Attention}(\text{LN}(\mathbf{x}))$$
$$\mathbf{x}'' = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$$

Normalisation is applied to the sub-layer input before passing it to attention or FFN.

**Key structural difference.** In pre-norm, the residual stream $\mathbf{x}$ bypasses all normalisation. Gradients flow from the output directly through the residual path without passing through any normalisation operation.

---

## Intermediate

### Q4. Why does pre-norm train more stably than post-norm for deep networks?

**Answer.**

**The gradient flow argument.**

In post-norm, the gradient through layer $\ell$ is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\ell+1}} \cdot \frac{\partial \text{LN}(\mathbf{x}_\ell + F(\mathbf{x}_\ell))}{\partial \mathbf{x}_\ell}$$

The Jacobian $\frac{\partial \text{LN}}{\partial \mathbf{x}}$ depends on the variance of the input. At initialisation, $F(\mathbf{x}_\ell) \approx 0$ (small residual), so $\text{LN}$ normalises a sum dominated by $\mathbf{x}_\ell$. As the network deepens, if activations grow or shrink, the LN Jacobian can become very small (if input has large norm) or amplified (if input has small norm), causing gradient instability.

In pre-norm, the gradient splits at the residual junction:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\ell+1}} \cdot \left(I + \frac{\partial F(\text{LN}(\mathbf{x}_\ell))}{\partial \mathbf{x}_\ell}\right)$$

The identity matrix $I$ (from the residual path) is always present, regardless of $F$. This guarantees:

$$\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_\ell}\right\| \geq \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\ell+1}}\right\| \cdot (1 - \epsilon)$$

for small perturbations — gradients cannot vanish through the skip path. This allows larger learning rates and more stable convergence.

**Empirical consequences:**
- Post-norm requires careful warmup (gradual LR increase over 1000+ steps); pre-norm can often use a constant or cosine schedule immediately
- Post-norm can fail entirely without warmup for very deep networks ($> 48$ layers)
- Pre-norm converges reliably for $100+$ layer models

**The trade-off.** Pre-norm can cause representation degeneration at the final layer: since the last block's FFN output is added to the residual without normalisation, the final representation can have arbitrary scale. This is why most pre-norm models add a final LayerNorm after all blocks (the "final LN" in GPT-2, LLaMA, etc.).

---

### Q5. Why is a final LayerNorm needed in pre-norm models?

**Answer.**

In a pre-norm model, the output of the last block is:

$$\mathbf{x}_L = \mathbf{x}_{L-1} + \text{FFN}(\text{LN}(\mathbf{x}_{L-1} + \text{Attn}(\text{LN}(\mathbf{x}_{L-1}))))$$

The residual stream $\mathbf{x}_L$ can have arbitrary scale because it is the sum of all residual contributions across $L$ layers, none of which is normalised after their addition. The scale grows roughly as $O(\sqrt{L})$ if residuals are independent, or faster if there are systematic additive contributions.

When the vocabulary projection $W_{\text{unembed}} \cdot \mathbf{x}_L$ is computed, an unnormalised input causes the logit distribution to be at an arbitrary scale, making it harder for the temperature parameter to be calibrated and potentially causing numerical issues.

The final LayerNorm fixes this:

$$\text{logits} = W_{\text{unembed}} \cdot \text{LN}(\mathbf{x}_L)$$

This normalises the residual stream before the output projection, ensuring consistent logit scales regardless of depth or training stage.

---

### Q6. What is QK-norm and when is it used?

**Answer.**

QK-norm (Henry et al., 2020; used in Gemma 2, some vision transformers) applies a normalisation layer to the query and key vectors inside the attention computation:

$$Q_h' = \text{LN}(Q_h), \quad K_h' = \text{LN}(K_h)$$
$$\text{Attention} = \text{Softmax}\!\left(\frac{Q_h' K_h'^T}{\sqrt{d_k}}\right) V_h$$

Alternatively, RMSNorm is used instead of LayerNorm for efficiency.

**Why QK-norm helps training stability.**

The attention logits $Q K^T / \sqrt{d_k}$ can become very large in magnitude during training, causing the softmax to saturate (extreme values near 0 or 1). Saturated softmax produces near-zero gradients, causing "attention collapse" — a specific training instability where attention distributions become extremely peaked or flat, and the gradient signal is lost.

With QK-norm, queries and keys are normalised to unit variance before computing the dot product. The logits are bounded: $|Q_i \cdot K_j| \leq d_k$ after normalisation (by Cauchy-Schwarz), preventing extreme softmax values.

**When it is used:**

- **Gemma 2**: applies RMSNorm to Q and K for stable training at scale
- **Vision Transformers (ViT)**: QK-norm appears in several large ViT variants
- **High learning rate training**: QK-norm enables using larger learning rates without attention collapse
- **Long context**: at long context lengths, the attention logit magnitudes can grow due to the large number of key-query pairs; QK-norm mitigates this

**The cost.** Two additional normalisation operations per attention layer, each of cost $O(Nd_k)$ — small relative to the attention computation itself.

---

## Advanced

### Q7. Analyse the backward pass of RMSNorm vs LayerNorm. What is the computational saving?

**Answer.**

**LayerNorm backward pass.**

Forward: $y = (x - \mu) / \sigma \cdot \gamma + \beta$

Given $\partial \mathcal{L} / \partial y$, the gradient with respect to $x$ requires:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{\sigma} \left[\frac{\partial \mathcal{L}}{\partial y_i} - \frac{1}{d}\sum_j \frac{\partial \mathcal{L}}{\partial y_j} - \frac{x_i - \mu}{\sigma^2 d} \sum_j \frac{\partial \mathcal{L}}{\partial y_j}(x_j - \mu)\right]$$

This requires:
- Computing $\sum_j \frac{\partial \mathcal{L}}{\partial y_j}$ (one reduction over dimension $d$)
- Computing $\sum_j \frac{\partial \mathcal{L}}{\partial y_j}(x_j - \mu)$ (another reduction, requiring $\mu$ from the forward pass)
- Two reductions total, plus elementwise operations

**RMSNorm backward pass.**

Forward: $y = x / \text{RMS}(x) \cdot \gamma$

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{\text{RMS}(x)} \left[\frac{\partial \mathcal{L}}{\partial y_i} - \frac{x_i}{d \cdot \text{RMS}(x)^2} \sum_j \frac{\partial \mathcal{L}}{\partial y_j} \gamma_j x_j / \text{RMS}(x)\right]$$

This requires:
- Computing $\sum_j \frac{\partial \mathcal{L}}{\partial y_j} \gamma_j x_j$ (one reduction)
- One reduction total (vs two for LayerNorm)
- No mean computation, no mean-centring

**Summary.** RMSNorm backward requires one fewer reduction over the $d$-dimensional feature vector. For $d = 4096$ and batch $\times$ sequence size of $B \cdot T$, this is a meaningful saving. Empirically: $10\text{--}15\%$ faster normalisation in both forward and backward passes.

---

### Q8. What is the effect of initialisation on pre-norm vs post-norm training dynamics?

**Answer.**

**At initialisation, all sub-layers $F(\mathbf{x}) \approx 0$ (small random outputs).**

**Post-norm at init:** The residual is $\mathbf{x}_\ell + F(\mathbf{x}_\ell) \approx \mathbf{x}_\ell$. The LayerNorm normalises $\mathbf{x}_\ell$, which is already near-unit-norm at initialisation. Everything is stable. The problem emerges as training progresses: $F$ grows, and the normalisation has to "fight" against growing residuals, causing unstable dynamics.

**Pre-norm at init:** Each sub-layer receives a normalised input $\text{LN}(\mathbf{x}_\ell)$ and produces a small output (near zero). The residual connection adds a small perturbation to $\mathbf{x}_\ell$. The norms of intermediate representations grow slowly and predictably.

**The "rank collapse" issue with pre-norm.** As training progresses in deep pre-norm models, the residual stream can become dominated by a few directions (the sum of many residual updates). The final representations have lower effective rank than desired. Solutions:
- Final LayerNorm (as discussed above)
- Scaled residuals: $\mathbf{x}' = \mathbf{x} + \alpha \cdot F(\text{LN}(\mathbf{x}))$ with $\alpha < 1$ (used in some large models)

**Gemma 2's post-MLP LayerNorm.** Applying RMSNorm after each sub-layer output (before adding to the residual) addresses both: it bounds the magnitude of residual contributions while still using pre-norm for the sub-layer inputs. This is the "pre-norm + post-norm" hybrid:

$$\mathbf{x}' = \mathbf{x} + \text{LN}_2(\text{Attention}(\text{LN}_1(\mathbf{x})))$$

The inner $\text{LN}_1$ stabilises the attention input; the outer $\text{LN}_2$ bounds the magnitude of what is added to the residual.
