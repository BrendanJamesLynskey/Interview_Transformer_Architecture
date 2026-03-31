# RoPE and ALiBi: Position Encodings

## Overview

Position encoding is a fundamental design choice: transformers have no inherent notion of token order, so positional information must be injected. This file covers the two dominant modern approaches — Rotary Position Embeddings (RoPE) and Attention with Linear Biases (ALiBi) — plus comparison with classical methods.

---

## Fundamentals

### Q1. Why do transformers need explicit position encodings?

**Answer.**

The self-attention operation is permutation-equivariant: if you shuffle the input tokens, the output is shuffled in the same way, but each output vector is identical to what it would have been at that position. There is no mechanism to distinguish "the word at position 3" from "the word at position 7" based on attention alone.

Formally, let $\pi$ be any permutation. For the attention function $f$:
$$f(\mathbf{x}_{\pi(1)}, \ldots, \mathbf{x}_{\pi(n)}) = \pi\!\left(f(\mathbf{x}_1, \ldots, \mathbf{x}_n)\right)$$

The output is just a permutation of what it would have been in the original order — each output vector is unaffected by order. This is desirable for set-processing tasks but catastrophic for language, where word order is essential. Position encodings break this symmetry.

---

### Q2. What are sinusoidal position encodings and what are their properties?

**Answer.**

The original Transformer (Vaswani et al., 2017) uses fixed sinusoidal encodings added directly to token embeddings:

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

where $pos$ is the token position and $i \in \{0, 1, \ldots, d/2 - 1\}$ is the dimension pair index.

**Properties:**
- Each position produces a unique vector (different $pos$ values are distinct)
- The shift property: $PE(pos + k)$ can be expressed as a fixed linear transformation of $PE(pos)$, regardless of $pos$. This is why the model can potentially infer relative positions
- Low dimension indices ($i$ small): high frequencies — the sine completes a full cycle over $\sim 6$ positions, encoding fine-grained local structure
- High dimension indices ($i$ large): low frequencies — the sine cycles slowly (over thousands of positions), encoding coarse global structure

**Limitations:**
- Absolute encodings: relative position is not directly available; the model must learn to infer $pos_1 - pos_2$ from $PE(pos_1)$ and $PE(pos_2)$
- The positional signal is mixed into the token embedding additively, potentially interfering with semantic content
- Extrapolation is poor in practice despite the function being defined for all $pos$ — the model learns to associate specific activation patterns with specific positions, so unseen large positions cause out-of-distribution behaviour

---

### Q3. What are learned absolute position embeddings and how do they compare to sinusoidal?

**Answer.**

Learned absolute embeddings (GPT-2, BERT) treat position as a categorical variable. A parameter matrix $P \in \mathbb{R}^{T_{\max} \times d}$ is trained end-to-end; the embedding for position $pos$ is row $P[pos]$, added to the token embedding.

**Key comparison:**

| Property | Sinusoidal | Learned |
|---|---|---|
| Parameters | None (fixed formula) | $T_{\max} \times d$ |
| Within-training quality | Slightly lower | Slightly higher |
| Extrapolation | Defined but poor in practice | Hard cutoff at $T_{\max}$ |
| Positional structure | Geometric (multi-frequency) | Arbitrary (data-driven) |
| Transferability | Better (no position overfitting) | Worse |

The practical conclusion: both fail badly beyond $T_{\max}$. Neither encodes relative position directly. These limitations motivated RoPE and ALiBi.

---

## Intermediate

### Q4. Derive the RoPE rotation matrix formulation. How does it encode relative position?

**Answer.**

RoPE (Su et al., 2021) encodes position by rotating query and key vectors such that their dot product depends only on the content vectors and their relative position.

**Desired property.** Find a function $f(\mathbf{x}, m)$ such that:
$$\langle f(\mathbf{q}, m),\, f(\mathbf{k}, n) \rangle = g(\mathbf{q}, \mathbf{k}, m - n)$$

**2D derivation via complex numbers.** Treat the 2D vector as a complex number: $z = x_1 + i x_2$. Define:
$$f(z, pos) = z \cdot e^{i \cdot pos \cdot \theta}$$

Then the inner product (real part of the conjugate product):
$$\text{Re}\!\left[f(q, m) \cdot \overline{f(k, n)}\right] = \text{Re}\!\left[q e^{im\theta} \cdot \bar{k} e^{-in\theta}\right] = \text{Re}\!\left[q\bar{k} \cdot e^{i(m-n)\theta}\right]$$

This depends only on $q$, $k$, and $(m-n)$ — exactly the desired property.

**Matrix form (2D).** Multiplication by $e^{i \cdot pos \cdot \theta}$ corresponds to the rotation matrix:

$$R(\theta, pos) = \begin{pmatrix} \cos(pos \cdot \theta) & -\sin(pos \cdot \theta) \\ \sin(pos \cdot \theta) & \cos(pos \cdot \theta) \end{pmatrix}$$

**Full $d$-dimensional formulation.** For $d$-dimensional vectors (with $d$ even), pair adjacent dimensions $(2i, 2i+1)$ and apply an independent rotation with frequency:

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1$$

The full rotation matrix is block-diagonal:

$$R_{\Theta, pos} = \begin{pmatrix} R(\theta_0, pos) & & \\ & \ddots & \\ & & R(\theta_{d/2-1}, pos) \end{pmatrix}$$

**Relative position in attention.** The attention logit between position $m$ and position $n$:

$$(R_{\Theta,m}\, \mathbf{q})^T (R_{\Theta,n}\, \mathbf{k}) = \mathbf{q}^T R_{\Theta,m}^T R_{\Theta,n}\, \mathbf{k} = \mathbf{q}^T R_{\Theta, n-m}\, \mathbf{k}$$

(See worked problem `problem_01_rope_rotation_matrices.md` for the algebraic proof that $R_{\Theta,m}^T R_{\Theta,n} = R_{\Theta,n-m}$.)

**Efficient computation.** In practice, the rotation is applied element-wise without materialising the matrix:

$$\tilde{q}_{2i} = q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i)$$
$$\tilde{q}_{2i+1} = q_{2i+1}\cos(m\theta_i) + q_{2i}\sin(m\theta_i)$$

This is equivalent to:
$$\tilde{\mathbf{q}} = \mathbf{q} \odot \cos(m\boldsymbol{\theta}) + \mathbf{q}^{\perp} \odot \sin(m\boldsymbol{\theta})$$

where $\mathbf{q}^{\perp}$ is $\mathbf{q}$ with each adjacent pair $(q_{2i}, q_{2i+1})$ replaced by $(-q_{2i+1}, q_{2i})$.

---

### Q5. What are the key properties of RoPE that make it well-suited for LLMs?

**Answer.**

**1. Relative position encoding by construction.**
The attention score between positions $m$ and $n$ depends only on $m - n$ (not on $m$ or $n$ individually). The model learns to attend based on distance, which is the natural inductive bias for language.

**2. Decaying bias with distance.**
The inner product $\mathbf{q}^T R_{\Theta, n-m} \mathbf{k}$ tends to decrease as $|m-n|$ increases, because the rotation causes vectors to become less aligned on average. This provides an automatic local-attention bias without any explicit penalty.

**3. Zero additional parameters.**
RoPE is a deterministic transformation; no learnable positional parameters exist. This is memory-efficient and eliminates the risk of position overfitting.

**4. Compatible with KV caching.**
RoPE modifies only $Q$ and $K$, not the residual stream. Cached keys are already rotated by their position at cache time. At decode step $t$, the new query is rotated by $\theta_t$ and attends to all cached keys rotated by their respective positions — the relative rotation automatically produces correct relative-position attention.

**5. Context extension via rescaling.**
Because RoPE is a mathematical function of position, several principled extension methods exist:

- **Linear interpolation (scale factor $s$):** $pos' = pos / s$ for $s > 1$. Interpolates positions into the trained range. Requires brief fine-tuning.
- **NTK-aware interpolation:** Replace base $b = 10000$ with $b' = b \cdot s^{d/(d-2)}$. Distributes the interpolation across frequency bands more intelligently than linear scaling.
- **YaRN:** Applies interpolation only to high-frequency dimensions (which extrapolate poorly) and leaves low-frequency dimensions unmodified. Achieves the best perplexity at extended contexts.

---

### Q6. Describe ALiBi. How does it produce relative position information?

**Answer.**

ALiBi (Attention with Linear Biases; Press et al., 2022) adds a fixed linear distance penalty to attention logits before softmax:

$$a_{ij} = \frac{q_i k_j^T}{\sqrt{d_k}} - m_h \cdot |i - j|$$

where $m_h$ is a head-specific slope (no gradient, no learnable parameters).

**Slope schedule.** For $H$ heads, the slopes are geometrically spaced:
$$m_h = 2^{-8h/H}, \quad h = 1, \ldots, H$$

For $H = 8$: slopes $= \{0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625\}$

- Head with slope $0.5$: attention decays sharply, effectively attends only to the last $\sim 10$ tokens
- Head with slope $0.004$: attention decays slowly, can integrate information over hundreds of tokens

This multi-scale structure allows the model to use different heads for local and global context.

**Why no learned parameters are needed.** The penalty $-m_h \cdot |i-j|$ is a monotone function of distance, so the model can learn a consistent interpretation: "more negative logit bias = further away." This is a strong, simple inductive bias that does not need to be discovered from data.

---

### Q7. Compare RoPE and ALiBi across the key properties that matter in practice.

**Answer.**

| Property | Sinusoidal | Learned | RoPE | ALiBi |
|---|---|---|---|---|
| Encodes relative position | Indirectly | Indirectly | Yes (by construction) | Yes (additive bias) |
| Learnable parameters | None | $T_{\max} \times d$ | None | None |
| Where applied | Added to embedding | Added to embedding | Rotates Q, K | Bias on logits |
| Within-training perplexity | Good | Best | Best | Slightly lower |
| Extrapolation (no fine-tuning) | Poor | None | Moderate | Good |
| Extrapolation (with fine-tuning) | N/A | N/A | Excellent (YaRN) | Good |
| KV cache compatible | Yes | Yes | Yes | Yes |
| Implementation complexity | Low | Lowest | Medium | Low |
| Widely adopted | Historical | Historical | Yes (most LLMs) | Limited |

**Key trade-off:** RoPE achieves better perplexity at and within the training length due to its richer, content-aware positional encoding. ALiBi extrapolates more gracefully out of the box because the linear bias is a well-defined function for any distance.

In practice, RoPE with YaRN or NTK-aware interpolation matches or surpasses ALiBi at extended contexts while maintaining better in-distribution quality. This is why RoPE has become dominant in modern LLMs.

---

## Advanced

### Q8. Prove that $R_{\Theta,m}^T R_{\Theta,n} = R_{\Theta, n-m}$.

**Answer.**

Since $R_{\Theta, m}$ is block-diagonal with blocks $R(\theta_i, m)$, it suffices to prove the identity for a single 2D rotation block.

**Claim.** $R(\theta, m)^T R(\theta, n) = R(\theta, n-m)$

**Proof.**

$$R(\theta, m)^T = \begin{pmatrix} \cos m\theta & \sin m\theta \\ -\sin m\theta & \cos m\theta \end{pmatrix} \quad \text{(transpose of rotation = inverse rotation)}$$

$$R(\theta, m)^T R(\theta, n) = \begin{pmatrix} \cos m\theta & \sin m\theta \\ -\sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} \cos n\theta & -\sin n\theta \\ \sin n\theta & \cos n\theta \end{pmatrix}$$

Top-left entry: $\cos m\theta \cos n\theta + \sin m\theta \sin n\theta = \cos(n\theta - m\theta) = \cos(n-m)\theta$

Top-right entry: $-\cos m\theta \sin n\theta + \sin m\theta \cos n\theta = \sin(m\theta - n\theta) = -\sin(n-m)\theta$

Bottom-left entry: $-\sin m\theta \cos n\theta + \cos m\theta \sin n\theta = \sin(n\theta - m\theta) = \sin(n-m)\theta$

Bottom-right entry: $\sin m\theta \sin n\theta + \cos m\theta \cos n\theta = \cos(n-m)\theta$

Therefore:
$$R(\theta, m)^T R(\theta, n) = \begin{pmatrix} \cos(n-m)\theta & -\sin(n-m)\theta \\ \sin(n-m)\theta & \cos(n-m)\theta \end{pmatrix} = R(\theta, n-m) \qquad \blacksquare$$

The full result follows because $R_{\Theta, m}^T R_{\Theta, n}$ is block-diagonal with blocks $R(\theta_i, m)^T R(\theta_i, n) = R(\theta_i, n-m)$, which assembled is $R_{\Theta, n-m}$.

---

### Q9. What is NTK-aware RoPE interpolation and why is it preferable to linear interpolation?

**Answer.**

**The problem.** When extending context from training length $L$ to $L' = s \cdot L$, positions $pos > L$ cause out-of-distribution rotation angles. Linear interpolation fixes this by remapping: $pos \to pos/s$, ensuring all rotation angles $pos \cdot \theta_i / s$ remain in the trained range $[0, L \cdot \theta_i]$.

**Why linear interpolation fails at high frequency.** The high-frequency dimensions (small $i$, large $\theta_i = 10000^{-2i/d} \approx 1$ for $i = 0$) already have very fine-grained positional discrimination at training time. Dividing by $s$ compresses these already-dense representations further, causing nearby tokens to have nearly identical high-frequency components. The model loses the ability to distinguish adjacent positions.

**NTK-aware interpolation.** Based on the neural tangent kernel view that a network generalises to inputs of similar frequency content as training. The key insight: increase the base $b = 10000$ to spread frequencies:

$$b' = b \cdot s^{d/(d-2)}$$

**Effect on $\theta_i$.** New frequencies:
$$\theta_i' = (b')^{-2i/d} = b^{-2i/d} \cdot s^{-2i/(d-2)}$$

- For $i = 0$ (highest frequency): $\theta_0' \approx \theta_0 / s^{2/(d-2)} \approx \theta_0$ (barely changes)
- For $i = d/2 - 1$ (lowest frequency): $\theta_{d/2-1}' \approx \theta_{d/2-1} \cdot s^{-(d-2)/(d-2)} = \theta_{d/2-1} / s$ (full linear interpolation)

High-frequency dimensions are left approximately unchanged (they extrapolate well because their angles are always small for nearby positions). Low-frequency dimensions receive the full interpolation. This is the optimal frequency-dependent trade-off.

**Practical outcome.** NTK-aware interpolation achieves:
- Better perplexity than linear interpolation at extended contexts, especially without fine-tuning
- The ability to extend by $4\text{-}16\times$ with short fine-tuning ($\sim 1000$ steps on long documents)
