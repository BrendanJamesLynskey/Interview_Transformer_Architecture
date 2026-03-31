# Worked Problem 02: Parameter Counting in a Transformer

**Difficulty:** Intermediate — expected in system design and ML engineering interviews.

**Skills tested:** Ability to derive parameter counts from first principles, numerical fluency with large-scale model dimensions, understanding of parameter distribution across components.

---

## Problem Statement

**(a)** Derive a general formula for the total number of parameters in a decoder-only Transformer as a function of:
- $d_{\text{model}}$: model dimension
- $n_h$: number of attention heads
- $d_{ff}$: feed-forward dimension (typically $4 d_{\text{model}}$)
- $L$: number of layers
- $V$: vocabulary size

Assume: Pre-LN with RMSNorm, no biases in attention/FFN, SwiGLU FFN (3 weight matrices), GQA with $g$ KV head groups, and tied input/output embeddings.

**(b)** Apply the formula to GPT-2 Small and derive the oft-cited "117M" parameter count.

**(c)** Apply the formula to GPT-3 and verify the "175B" parameter count.

**(d)** What fraction of parameters in GPT-3 are in the attention sub-layers vs. FFN sub-layers vs. embeddings?

---

## Part (a): General Parameter Formula

### Token Embeddings

$$P_{\text{embed}} = V \times d_{\text{model}}$$

With **tied input/output embeddings**, the output projection matrix is the transpose of the embedding matrix — no additional parameters.

Without tying: $P_{\text{embed}} = 2 \times V \times d_{\text{model}}$.

### Per-Layer Parameters

#### Multi-Head Attention (with GQA)

Let $d_h = d_{\text{model}} / n_h$ (dimension per head).

- $W^Q$: $n_h$ query heads, each $d_{\text{model}} \times d_h$ → combined: $d_{\text{model}} \times d_{\text{model}}$
- $W^K$: $g$ KV groups, each $d_{\text{model}} \times d_h$ → combined: $d_{\text{model}} \times (g \cdot d_h)$
- $W^V$: $g$ KV groups → combined: $d_{\text{model}} \times (g \cdot d_h)$
- $W^O$: $(n_h \cdot d_h) \times d_{\text{model}} = d_{\text{model}} \times d_{\text{model}}$

$$P_{\text{attn}} = d_{\text{model}}^2 + 2 \cdot d_{\text{model}} \cdot g \cdot d_h + d_{\text{model}}^2 = 2d_{\text{model}}^2 + 2 d_{\text{model}} \cdot g \cdot \frac{d_{\text{model}}}{n_h}$$

For **full MHA** ($g = n_h$): $P_{\text{attn}} = 4 d_{\text{model}}^2$

For **MQA** ($g = 1$): $P_{\text{attn}} = 2d_{\text{model}}^2 + \frac{2d_{\text{model}}^2}{n_h}$

#### SwiGLU Feed-Forward Network

SwiGLU uses **three** weight matrices (instead of the standard two):

$$\text{FFN}(x) = (\text{SiLU}(xW_1) \odot xW_3) W_2$$

where $W_1, W_3 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ and $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$.

Note: When SwiGLU is used, $d_{ff}$ is typically set to $\frac{2}{3} \times 4 d_{\text{model}} \approx 2.67 d_{\text{model}}$ so the total FFN parameters match the standard two-matrix version.

$$P_{\text{ffn}} = 2 \times (d_{\text{model}} \times d_{ff}) + (d_{ff} \times d_{\text{model}}) = 3 \times d_{\text{model}} \times d_{ff}$$

With standard $d_{ff} = 4 d_{\text{model}}$: $P_{\text{ffn}} = 12 d_{\text{model}}^2$

#### RMSNorm

Two RMSNorm layers per Transformer layer (one before attention, one before FFN):

$$P_{\text{norm}} = 2 \times d_{\text{model}}$$

(Small and often excluded from estimates.)

### Total Formula

$$\boxed{P_{\text{total}} = \underbrace{V \cdot d_{\text{model}}}_{\text{embeddings}} + L \cdot \left(\underbrace{P_{\text{attn}}}_{\text{attention}} + \underbrace{3 d_{\text{model}} d_{ff}}_{\text{FFN}} + \underbrace{2 d_{\text{model}}}_{\text{RMSNorm}}\right)}$$

For the common case (MHA, standard FFN, negligible norm):

$$P_{\text{total}} \approx V \cdot d_{\text{model}} + L \cdot (4 d_{\text{model}}^2 + 8 d_{\text{model}}^2) = V \cdot d_{\text{model}} + 12 L \cdot d_{\text{model}}^2$$

---

## Part (b): GPT-2 Small (117M Parameters)

**Architecture:**

| Hyperparameter | Value |
|---|---|
| $d_{\text{model}}$ | 768 |
| $n_h$ (heads) | 12 |
| $d_{ff}$ | 3,072 ($= 4 \times 768$) |
| $L$ (layers) | 12 |
| $V$ (vocab size) | 50,257 |
| Biases | Yes (GPT-2 uses biases) |
| Position embeddings | Learned, separate matrix |
| Tied embeddings | Yes (input = output) |

**GPT-2 uses standard ReLU/GeLU, not SwiGLU, and includes biases. Adjust formulas accordingly.**

#### Token Embeddings

$$P_{\text{token\_embed}} = 50{,}257 \times 768 = 38{,}597{,}376 \approx 38.6\text{M}$$

#### Position Embeddings (GPT-2 uses absolute learned PE)

$$P_{\text{pos\_embed}} = 1{,}024 \times 768 = 786{,}432 \approx 0.8\text{M}$$

(Max context length 1,024; separate learnable matrix.)

#### Per Attention Layer (MHA, with biases)

With biases, each projection adds $d_k$ or $d_{\text{model}}$ extra terms:

- $W^Q, W^K, W^V$: each $768 \times 768 + 768 = 590{,}592$ → 3 projections: $1{,}771{,}776$
- $W^O$: $768 \times 768 + 768 = 590{,}592$
- **Attention total per layer:** $2{,}362{,}368$

#### Per FFN Layer (with biases)

- $W_1$: $768 \times 3072 + 3072 = 2{,}362{,}368$
- $W_2$: $3072 \times 768 + 768 = 2{,}360{,}064$
- **FFN total per layer:** $4{,}722{,}432$

#### LayerNorm (GPT-2 uses LayerNorm)

Two per layer: $2 \times (768 + 768) = 3{,}072$ (scale + shift)

#### Final LayerNorm (after last layer)

$768 + 768 = 1{,}536$

#### Per-Layer Total

$$P_{\text{per layer}} = 2{,}362{,}368 + 4{,}722{,}432 + 3{,}072 = 7{,}087{,}872$$

#### Total for 12 Layers

$$12 \times 7{,}087{,}872 = 85{,}054{,}464 \approx 85.1\text{M}$$

#### Grand Total

$$P_{\text{total}} = 38.6\text{M} + 0.8\text{M} + 85.1\text{M} + 0.002\text{M (final LN)} \approx 124.5\text{M}$$

**Wait — where does "117M" come from?**

The discrepancy arises from:
1. GPT-2's output layer **is not tied** to the input embeddings in the original implementation — but the "117M" figure counts non-embedding parameters by convention.
2. Different sources count parameters differently (include/exclude embeddings, include/exclude biases).

**Non-embedding parameter count:**

$$P_{\text{non-embed}} = 12 \times 7{,}087{,}872 + 1{,}536 \approx 85\text{M}$$

**The "117M" typically includes both embedding tables but the figure varies by source.** The exact Hugging Face count for `gpt2` is **124,439,808** parameters (with both embedding tables), or **117,653,760** if the output projection reuses the embedding weights.

---

## Part (c): GPT-3 (175B Parameters)

**Architecture:**

| Hyperparameter | Value |
|---|---|
| $d_{\text{model}}$ | 12,288 |
| $n_h$ (heads) | 96 |
| $d_{ff}$ | 49,152 ($= 4 \times 12{,}288$) |
| $L$ (layers) | 96 |
| $V$ (vocab size) | 50,257 |
| Biases | Yes |
| Tied embeddings | No |

**Applying the formula** $P \approx V \cdot d + 12 L d^2$ (approximate, ignoring biases):

$$P_{\text{embed}} = 50{,}257 \times 12{,}288 \approx 617\text{M}$$

$$P_{\text{layers}} = 12 \times 96 \times 12{,}288^2 = 12 \times 96 \times 150{,}994{,}944 \approx 174.0\text{B}$$

$$P_{\text{total}} \approx 174.0\text{B} + 0.617\text{B} \approx 174.6\text{B}$$

**With the separate output embedding matrix:**

$$P_{\text{total}} \approx 2 \times 0.617\text{B} + 174.0\text{B} \approx 175.2\text{B}$$

This matches the published "175B" figure.

---

## Part (d): Parameter Distribution in GPT-3

**Detailed per-layer breakdown:**

Per layer, MHA:

$$P_{\text{attn}} = 4 \times 12{,}288^2 = 4 \times 150{,}994{,}944 \approx 604\text{M}$$

Per layer, FFN:

$$P_{\text{ffn}} = 2 \times 12{,}288 \times 49{,}152 = 8 \times 12{,}288^2 \approx 1{,}208\text{M}$$

**Total across all 96 layers:**

| Component | Parameters | Fraction |
|---|---|---|
| Token embeddings (in + out) | $\approx 1.234\text{B}$ | 0.7% |
| Attention (96 layers) | $96 \times 604\text{M} = 58.0\text{B}$ | 33.1% |
| FFN (96 layers) | $96 \times 1{,}208\text{M} = 116.0\text{B}$ | 66.3% |
| **Total** | $\approx 175.2\text{B}$ | **100%** |

**Key insight:** The feed-forward network contains approximately **twice as many parameters as the attention layers** in a standard Transformer. This is because:

$$\frac{P_{\text{ffn}}}{P_{\text{attn}}} = \frac{8 d_{\text{model}}^2}{4 d_{\text{model}}^2} = 2$$

The FFN accounts for roughly 2/3 of all layer parameters. This is why:
- Parameter-efficient fine-tuning (LoRA) often applies to both attention and FFN
- Mixture-of-Experts (MoE) architectures typically replace the FFN (not attention) with sparse experts — this is where the parameters are
- Knowledge/facts seem to be stored in FFN layers (Geva et al., 2021)

---

## Summary: Useful Rules of Thumb

Given $d_{\text{model}} = d$ and $L$ layers:

$$P_{\text{total}} \approx 12 L d^2 \quad \text{(for } V d \ll 12 L d^2 \text{)}$$

This approximation is accurate to within a few percent for large models. Rearranging:

$$d_{\text{model}} \approx \sqrt{\frac{P_{\text{total}}}{12 L}}$$

**Examples:**

| Model | $P$ | $L$ | Formula prediction $d$ | Actual $d$ |
|---|---|---|---|---|
| GPT-2 Small | 117M | 12 | $\sqrt{117\text{M} / 144} \approx 901$ | 768 |
| GPT-2 Large | 774M | 36 | $\sqrt{774\text{M} / 432} \approx 1{,}338$ | 1,280 |
| GPT-3 | 175B | 96 | $\sqrt{175\text{B} / 1{,}152} \approx 12{,}325$ | 12,288 |

The approximation underestimates $d$ slightly because it ignores embeddings, but it gives a strong first estimate.
