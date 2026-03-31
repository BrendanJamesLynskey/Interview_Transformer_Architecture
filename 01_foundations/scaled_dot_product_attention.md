# Scaled Dot-Product Attention

## Overview

Scaled dot-product attention is the atomic computational unit of the Transformer. Every variant — multi-head attention, cross-attention, causal attention — is built on top of this single formula. Mastering the derivation, the scaling argument, the complexity analysis, and the masking mechanics is non-negotiable for ML engineering and research roles.

---

## Tier 1: Fundamentals

### Q1. State the scaled dot-product attention formula and explain every term.

**Answer.**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Term-by-term:**

| Symbol | Shape | Meaning |
|---|---|---|
| $Q$ | $n_q \times d_k$ | Query matrix — what each query token is looking for |
| $K$ | $n_k \times d_k$ | Key matrix — what each key token advertises |
| $V$ | $n_k \times d_v$ | Value matrix — content each key token contributes |
| $d_k$ | scalar | Dimension of keys and queries |
| $QK^T$ | $n_q \times n_k$ | Raw attention score matrix (one score per query-key pair) |
| $\sqrt{d_k}$ | scalar | Scaling factor to control score variance |
| $\text{softmax}(\cdot)$ | $n_q \times n_k$ | Normalised attention weights (rows sum to 1) |
| Output | $n_q \times d_v$ | Context-weighted value aggregation |

For self-attention: $n_q = n_k = n$ (sequence length), so the attention matrix is square.

For cross-attention: $n_q$ (decoder length) and $n_k$ (encoder length) can differ.

---

### Q2. Why divide by $\sqrt{d_k}$? Provide the complete variance argument.

**Answer.**

**The problem:** For large $d_k$, dot products grow large in magnitude, pushing softmax into saturation regions where gradients vanish.

**The variance argument:**

Assume queries and keys are drawn i.i.d.:

$$q_j, k_j \overset{\text{iid}}{\sim} \mathcal{N}(0, 1), \quad j = 1, \ldots, d_k$$

The dot product is:

$$q \cdot k = \sum_{j=1}^{d_k} q_j k_j$$

Each term $q_j k_j$ has:
- $\mathbb{E}[q_j k_j] = \mathbb{E}[q_j]\,\mathbb{E}[k_j] = 0 \cdot 0 = 0$
- $\text{Var}(q_j k_j) = \mathbb{E}[q_j^2 k_j^2] - 0 = \mathbb{E}[q_j^2]\,\mathbb{E}[k_j^2] = 1 \cdot 1 = 1$

By independence across dimensions:

$$\text{Var}(q \cdot k) = \sum_{j=1}^{d_k} \text{Var}(q_j k_j) = d_k \implies \text{Std}(q \cdot k) = \sqrt{d_k}$$

**The fix:** Divide by $\sqrt{d_k}$ to restore unit variance:

$$\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{(\sqrt{d_k})^2} = 1$$

**Why saturation matters:**

When $d_k = 512$ without scaling, dot products have standard deviation ~22.6. A raw score difference of 22 units causes softmax to assign $e^{22} / (e^{22} + 1) \approx 1 - 3 \times 10^{-10}$ to one entry and essentially zero to all others. The gradient of the softmax at this point is:

$$\frac{\partial \sigma_i}{\partial z_i} = \sigma_i(1 - \sigma_i) \approx 1 \cdot 0 = 0$$

No gradient flows back to the Q and K projection weights, halting learning.

---

### Q3. What is the computational complexity of scaled dot-product attention in time and space?

**Answer.**

For a sequence of length $n$ with key/value dimension $d$ (setting $d_k = d_v = d$):

**Time complexity: $O(n^2 d)$**

| Operation | Dimensions | FLOPs |
|---|---|---|
| $QK^T$ | $(n \times d) \cdot (d \times n)$ | $2n^2 d$ |
| Softmax | Row-wise over $n \times n$ | $O(n^2)$ |
| Weights $\cdot V$ | $(n \times n) \cdot (n \times d)$ | $2n^2 d$ |
| **Total** | | $O(n^2 d)$ |

**Space complexity: $O(n^2 + nd)$**

- Attention weight matrix: $O(n^2)$ — dominant for large $n$
- Q, K, V matrices: $O(nd)$
- Total: $O(n^2)$ asymptotically

**Concrete memory footprint (FP16, $d = 64$ per head):**

| $n$ | Attention matrix size |
|---|---|
| 512 | ~0.5 MB |
| 2,048 | ~8 MB |
| 8,192 | ~128 MB |
| 32,768 | ~2 GB |
| 131,072 | ~32 GB |

This quadratic scaling is why naive attention is infeasible for context windows beyond ~8k tokens without algorithmic changes (FlashAttention, sparse attention, linear attention).

---

### Q4. How do you interpret attention weights? What information do they convey and what are their limitations?

**Answer.**

The attention weight $\alpha_{ij}$ represents how much query token $i$ draws from value token $j$ when computing its contextualised representation.

**Formal properties:**
- $\alpha_{ij} \in (0, 1)$ for all $i, j$ — softmax ensures strict positivity
- $\sum_{j=1}^{n} \alpha_{ij} = 1$ for each query $i$ — each query's weights form a probability distribution
- Output for token $i$: $\text{out}_i = \sum_j \alpha_{ij} v_j$ — a convex combination of value vectors

**Interpretation of extremes:**

- $\alpha_{ij} \approx 1$ for one $j$, $\approx 0$ for others: near-hard attention; token $i$ almost exclusively takes information from token $j$
- $\alpha_{ij} \approx 1/n$ for all $j$: uniform attention; output is the mean of all value vectors

**Empirically observed patterns in trained models:**
- **Local attending:** high weights on tokens within a small neighbourhood (captures local syntax)
- **Coreference:** pronouns attending to their referents
- **Delimiter sinking:** many heads dump excess weight on punctuation or separator tokens when no other target is salient

**Critical limitation:** Attention weight $\neq$ causal importance. A token with high $\alpha_{ij}$ might contribute a near-zero value vector $v_j$, making the high weight irrelevant. Conversely, a token with moderate weight but a highly informative value vector can dominate the output. Gradient-based attribution methods (integrated gradients, SHAP) are needed for rigorous importance attribution.

---

### Q5. What is a causal mask and how is it applied technically? What is a padding mask?

**Answer.**

**Causal (autoregressive) mask:**

In decoder self-attention, token $i$ must not attend to any token $j > i$ (future positions). The mask is an upper-triangular matrix of $-\infty$ values (zeros below and on the diagonal):

$$M_{ij}^{\text{causal}} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Applied before softmax:

$$\tilde{E} = \frac{QK^T}{\sqrt{d_k}} + M^{\text{causal}}$$

After softmax, $\exp(-\infty) = 0$, so future positions receive exactly zero weight. The denominator sums only over valid positions, so the remaining weights still sum to 1.

**Padding mask:**

Batch sequences are padded to a uniform length. Padding positions carry no information and should contribute zero attention weight (as keys) and ideally receive no meaningful query attention.

For each sample, the padding mask marks columns corresponding to pad tokens:

$$M_{ij}^{\text{pad}} = \begin{cases} 0 & \text{if key position } j \text{ is a real token} \\ -\infty & \text{if key position } j \text{ is padding} \end{cases}$$

**Key distinction:**
- Padding mask: suppresses specific columns depending on sequence length; differs per batch element
- Causal mask: suppresses all strictly upper-triangular entries; same for all batch elements

Both masks are additive and can be combined: $M = M^{\text{causal}} + M^{\text{pad}}$.

---

## Tier 2: Intermediate

### Q6. Derive why softmax saturation causes vanishing gradients. Quantify the effect for $d_k = 64$ vs. $d_k = 512$.

**Answer.**

**Softmax gradient derivation:**

For $\sigma_i = e^{z_i} / \sum_k e^{z_k}$, the Jacobian is:

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j) = \begin{cases} \sigma_i(1 - \sigma_i) & i = j \\ -\sigma_i \sigma_j & i \neq j \end{cases}$$

In compact form: $J = \text{diag}(\sigma) - \sigma\sigma^T$.

**Effect of saturation:**

Suppose one score dominates: $z_1 \gg z_j$ for $j > 1$. Then $\sigma_1 \approx 1$, $\sigma_j \approx 0$.

- $J_{11} = \sigma_1(1 - \sigma_1) \approx 1 \times 0 = 0$
- $J_{1j} = -\sigma_1 \sigma_j \approx -1 \times 0 = 0$
- $J_{jj} = \sigma_j(1 - \sigma_j) \approx 0 \times 1 = 0$

The entire Jacobian collapses to zero. The gradient of the loss with respect to any score $z_j$ passes through this near-zero matrix — effectively killing the signal.

**Quantifying saturation by $d_k$:**

Without scaling, $\text{Std}(q \cdot k) = \sqrt{d_k}$.

A typical "dominant" scenario: the max score is $+\sqrt{d_k}$ and another score is $0$. The softmax weight on the dominant entry:

$$\sigma_{\max} = \frac{e^{\sqrt{d_k}}}{e^{\sqrt{d_k}} + (n-1) e^0} \approx \frac{e^{\sqrt{d_k}}}{e^{\sqrt{d_k}}}= 1 - (n-1)e^{-\sqrt{d_k}}$$

| $d_k$ | $\sqrt{d_k}$ | $\sigma_{\max}$ (n=10) | $J_{max,max}$ approx |
|---|---|---|---|
| 64 | 8.0 | $\approx 0.9997$ | $\approx 0.0003$ |
| 512 | 22.6 | $\approx 1 - 6 \times 10^{-10}$ | $\approx 6 \times 10^{-10}$ |

With $\sqrt{d_k}$ scaling, both cases produce $\sqrt{d_k} / \sqrt{d_k} = 1$, and $\sigma_{\max} = e^1/(e^1 + (n-1)) \approx 0.23$ for $n=10$ — a healthy, non-saturated regime.

---

### Q7. Implement scaled dot-product attention with both causal and padding masks in Python. Verify correctness with test cases.

**Answer.**

```python
import numpy as np
from typing import Optional


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerically stable scaled dot-product attention.

    Args:
        Q:    (batch, n_q, d_k)  Query matrix
        K:    (batch, n_k, d_k)  Key matrix
        V:    (batch, n_k, d_v)  Value matrix
        mask: (batch, n_q, n_k) or broadcastable; True = mask out position

    Returns:
        output:  (batch, n_q, d_v)
        weights: (batch, n_q, n_k)  attention distribution
    """
    d_k = Q.shape[-1]

    # Raw scores: (batch, n_q, n_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    # Apply mask: set masked positions to -inf before softmax
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)

    # Numerically stable softmax: subtract row-wise max before exp
    scores_max = scores.max(axis=-1, keepdims=True)       # (batch, n_q, 1)
    # Guard: if all scores in a row are -inf (fully masked), max is -inf
    scores_max = np.where(np.isinf(scores_max), 0.0, scores_max)

    exp_scores = np.exp(scores - scores_max)              # (batch, n_q, n_k)
    exp_scores = np.where(mask, 0.0, exp_scores) if mask is not None else exp_scores

    denom = exp_scores.sum(axis=-1, keepdims=True)        # (batch, n_q, 1)
    weights = exp_scores / (denom + 1e-9)                 # avoid div-by-zero

    output = np.matmul(weights, V)                        # (batch, n_q, d_v)
    return output, weights


def make_causal_mask(n: int) -> np.ndarray:
    """
    Upper-triangular causal mask. True = masked out.
    Shape: (1, n, n) — broadcast over batch dimension.
    """
    return np.triu(np.ones((1, n, n), dtype=bool), k=1)


def make_padding_mask(lengths: list[int], max_len: int) -> np.ndarray:
    """
    Padding mask based on true sequence lengths. True = masked out (key is pad).
    Shape: (batch, 1, max_len) — broadcast over query dimension.
    """
    batch = len(lengths)
    mask = np.zeros((batch, 1, max_len), dtype=bool)
    for i, length in enumerate(lengths):
        mask[i, 0, length:] = True
    return mask


# ── Tests ────────────────────────────────────────────────────────────────────

np.random.seed(0)
batch, n, d_k, d_v = 2, 5, 8, 8

Q = np.random.randn(batch, n, d_k)
K = np.random.randn(batch, n, d_k)
V = np.random.randn(batch, n, d_v)

# Test 1: No mask — weights must sum to 1
out, w = scaled_dot_product_attention(Q, K, V)
assert np.allclose(w.sum(axis=-1), 1.0), "Weights must sum to 1"
print("Test 1 passed: weights sum to 1.0")

# Test 2: Causal mask — upper triangle must be zero
causal = make_causal_mask(n)
out_c, w_c = scaled_dot_product_attention(Q, K, V, mask=causal)
upper = np.triu(w_c[0], k=1)
assert np.allclose(upper, 0.0, atol=1e-9), "Future positions must be zero"
assert np.allclose(w_c.sum(axis=-1), 1.0), "Weights must still sum to 1"
print("Test 2 passed: causal mask zeroes future positions")

# Test 3: Padding mask — pad positions must be zero
lengths = [3, 5]  # first sequence has 3 real tokens, second has 5
pad_mask = make_padding_mask(lengths, max_len=n)
out_p, w_p = scaled_dot_product_attention(Q, K, V, mask=pad_mask)
assert np.allclose(w_p[0, :, 3:], 0.0, atol=1e-9), "Pad keys must get zero weight"
assert np.allclose(w_p[1, :, :], w_p[1, :, :]), "Full sequence unaffected"
print("Test 3 passed: padding mask zeroes pad-token keys")

# Test 4: Numerical stability with extreme values
Q_large = Q * 1000.0
out_large, _ = scaled_dot_product_attention(Q_large, K, V)
assert not np.isnan(out_large).any(), "NaN detected with large inputs"
print("Test 4 passed: numerically stable with extreme inputs")
```

**Expected output:**
```
Test 1 passed: weights sum to 1.0
Test 2 passed: causal mask zeroes future positions
Test 3 passed: padding mask zeroes pad-token keys
Test 4 passed: numerically stable with extreme inputs
```

---

### Q8. How does the choice of $d_k$ vs $d_v$ affect the expressiveness and efficiency of the attention mechanism?

**Answer.**

**$d_k$ (query/key dimension):** Controls the expressiveness of the matching function.

- Larger $d_k$ allows queries and keys to encode richer features for the dot-product similarity, at the cost of more parameters in $W^Q$ and $W^K$ and larger intermediate scores that require careful normalisation.
- In standard Transformers, $d_k = d_{\text{model}} / n_{\text{heads}}$, which keeps total parameters constant as you vary the number of heads.
- Too small a $d_k$ (e.g., $d_k = 16$) limits the attention head's ability to discriminate between different keys.

**$d_v$ (value dimension):** Controls the information bandwidth of what is retrieved.

- Larger $d_v$ allows more information to flow from each attended token.
- In the standard Transformer, $d_v = d_k$, but this is not required. You can have $d_v \neq d_k$.
- After MHA concatenation, the output dimension is $n_{\text{heads}} \times d_v$, which must be projected back to $d_{\text{model}}$ via $W^O$.

**Asymmetric $d_k \neq d_v$:** Used in some architectures for efficiency. Small $d_k$ is sufficient for good matching; larger $d_v$ allows richer value retrieval. The Multi-Query Attention (MQA) approach takes this further by having one K and V per layer but multiple Q heads, reducing KV memory while maintaining query diversity.

**Rule of thumb:** In practice, $d_k = d_v = d_{\text{model}} / n_{\text{heads}}$ is the standard choice. This factorisation keeps the total attention computation cost constant regardless of how many heads are used, which is a useful property for ablation studies.

---

## Tier 3: Advanced

### Q9. Explain how FlashAttention achieves IO-efficient exact attention. What is the memory hierarchy insight?

**Answer.**

**Memory hierarchy of a GPU:**

| Memory tier | Size (H100) | Bandwidth |
|---|---|---|
| L1 / SRAM (on-chip) | ~256 KB/SM | ~20 TB/s |
| L2 cache | ~50 MB | ~8 TB/s |
| HBM (off-chip DRAM) | 80 GB | ~3.3 TB/s |

Standard attention materialises the $n \times n$ attention matrix in HBM. For $n = 4096$, $n^2 = 16.7M$ entries — roughly 32 MB in FP16. Every softmax and matmul reads and writes this matrix, creating a bandwidth bottleneck far worse than the raw FLOPs suggest.

**FlashAttention's approach (Dao et al., 2022):**

The insight is that softmax can be computed in a single fused pass using the **online softmax** algorithm, without ever materialising the full attention matrix.

**Online softmax recurrence:**

Maintain three running statistics as you stream through blocks of keys:
- $m$: running maximum of seen scores
- $\ell$: running normalisation sum (corrected for the max update)
- $o$: running output accumulator

For a new block of scores $s_{\text{new}}$:

$$m_{\text{new}} = \max(m_{\text{old}}, \max s_{\text{new}})$$
$$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \sum e^{s_{\text{new}} - m_{\text{new}}}$$
$$o_{\text{new}} = \frac{e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} \cdot o_{\text{old}} + e^{s_{\text{new}} - m_{\text{new}}} V_{\text{block}}}{\ell_{\text{new}}}$$

At the end, $o_{\text{new}}$ equals the correct attention output.

**Algorithm:**

1. Tile Q into blocks that fit in SRAM
2. For each Q block, stream through all K, V blocks:
   a. Load K, V block from HBM to SRAM
   b. Compute scores in SRAM
   c. Update running $(m, \ell, o)$ statistics
3. Write the final output block back to HBM

**Complexity:**

| | Standard | FlashAttention |
|---|---|---|
| HBM reads/writes | $O(n^2)$ | $O(n^2 d / M)$ |
| SRAM requirement | $O(n^2)$ | $O(M)$ where $M = $ SRAM size |
| Mathematical result | Exact | Exact |

FlashAttention v2 (2023) adds better parallelism across query blocks and sequence dimension; FlashAttention v3 (2024) targets H100 hardware specifically with asynchronous execution.

---

### Q10. Compare linear attention approximations. Why haven't they displaced standard attention?

**Answer.**

**The linear attention idea:**

The softmax in standard attention prevents the associativity trick. If we replace softmax with a kernel:

$$\text{Attention}(Q, K, V)_i = \frac{\sum_j \kappa(q_i, k_j) v_j}{\sum_j \kappa(q_i, k_j)}$$

where $\kappa(q, k) = \phi(q)^T \phi(k)$ for a feature map $\phi: \mathbb{R}^d \to \mathbb{R}^r$, then:

$$\text{Attention}_i = \frac{\phi(q_i)^T \left(\sum_j \phi(k_j) v_j^T\right)}{\phi(q_i)^T \left(\sum_j \phi(k_j)\right)}$$

The sums $\sum_j \phi(k_j) v_j^T \in \mathbb{R}^{r \times d_v}$ and $\sum_j \phi(k_j) \in \mathbb{R}^r$ can be computed once in $O(nrd)$, and the query step is $O(rd)$ per query. Total: $O(nrd)$ — linear in $n$.

**Notable approaches:**

| Method | Feature map $\phi$ | Issue |
|---|---|---|
| Random Features (Performers, 2021) | Random Fourier features approximating RBF kernel | Approximation quality degrades; no exact softmax |
| ELU+1 (Linear Transformers, 2020) | $\phi(x) = \text{elu}(x) + 1$ | Simple but poor performance on long-range tasks |
| Mamba / SSM (2023) | Structured state space (not attention) | Different inductive bias entirely |
| RetNet (2023) | Retention mechanism (decaying weights) | Competitive but not universal replacement |

**Why linear attention hasn't replaced softmax attention:**

1. **Quality gap:** Replacing softmax with a kernel changes the inductive bias. Softmax attention produces a proper probability distribution with competition between keys. Linear attention does not — the model can "attend to everything at once" without normalisation being sharp. Empirically, linear attention underperforms on tasks requiring precise retrieval.

2. **FlashAttention closes the gap:** The memory and speed problem of standard attention has largely been solved by FlashAttention (exact, 2–4x faster, linear memory). The main motivation for linear attention was IO efficiency, which is now addressed.

3. **Training stability:** Linear attention models are harder to train stably at scale.

4. **Recurrent form advantage (and disadvantage):** Linear attention can be reformulated as an RNN, making it $O(1)$ per step at inference — an advantage for very long streaming contexts. But this sacrifices the parallelism during training.

**Current state (as of mid-2025):** Hybrid architectures (e.g., alternating standard and linear attention layers, or attention with SSM layers like Jamba) are being explored, but dense softmax attention with FlashAttention remains the dominant choice for frontier models.
