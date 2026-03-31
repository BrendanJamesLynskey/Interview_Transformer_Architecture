# Problem 01: RoPE Rotation Matrices

**Topic:** Rotary Position Embeddings — rotation matrix derivation and worked computation

**Difficulty:** Intermediate to Advanced

**Expected time:** 20–30 minutes

---

## Problem Statement

You are implementing RoPE (Rotary Position Embeddings) for a transformer with the following configuration:
- Hidden dimension $d = 8$ (for tractability)
- Head dimension $d_k = 8$ (single head)
- Base $b = 10000$
- Sequence positions of interest: $m = 2$ (query) and $n = 5$ (key)

**Part A.** Compute the rotation frequencies $\theta_i$ for $i = 0, 1, 2, 3$.

**Part B.** Write out the full $8 \times 8$ rotation matrix $R_{\Theta, m}$ for position $m = 2$.

**Part C.** Prove that the dot product $(R_{\Theta, m} \mathbf{q})^T (R_{\Theta, n} \mathbf{k})$ equals $\mathbf{q}^T R_{\Theta, n-m} \mathbf{k}$.

**Part D.** Given query vector $\mathbf{q} = [1, 0, 0, 1, 1, 0, 0, 1]^T$ at position $m = 2$ and key vector $\mathbf{k} = [1, 0, 0, 1, 1, 0, 0, 1]^T$ at position $n = 5$, compute:
1. The rotated vectors $\tilde{\mathbf{q}} = R_{\Theta,2}\,\mathbf{q}$ and $\tilde{\mathbf{k}} = R_{\Theta,5}\,\mathbf{k}$
2. The attention logit $\tilde{\mathbf{q}}^T \tilde{\mathbf{k}}$
3. Verify this equals $\mathbf{q}^T R_{\Theta, 3} \mathbf{k}$ (relative offset $n - m = 3$)

**Part E.** Explain intuitively: what happens to the attention logit as the relative position $|n - m|$ increases?

---

## Solution

### Part A: Rotation frequencies

For base $b = 10000$ and head dimension $d_k = 8$, the frequencies are:

$$\theta_i = b^{-2i/d_k} = 10000^{-2i/8} = 10000^{-i/4}, \quad i = 0, 1, 2, 3$$

| $i$ | $\theta_i = 10000^{-i/4}$ | Decimal value |
|---|---|---|
| 0 | $10000^{0} = 1$ | $1.000000$ |
| 1 | $10000^{-1/4} = 10^{-1} = 0.1$ | $0.100000$ |
| 2 | $10000^{-2/4} = 10^{-2} = 0.01$ | $0.010000$ |
| 3 | $10000^{-3/4} = 10^{-3} = 0.001$ | $0.001000$ |

**Intuition.** Dimension pair 0 rotates by angle $\theta_0 = 1$ radian per position — fast rotation encoding fine-grained local position. Dimension pair 3 rotates by $\theta_3 = 0.001$ radians per position — very slow rotation encoding coarse global position. This multi-scale structure is the positional analogue of a Fourier decomposition.

---

### Part B: Rotation matrix at position $m = 2$

The rotation angles at position $m = 2$:

$$m \theta_0 = 2 \times 1.0 = 2.0 \text{ rad}$$
$$m \theta_1 = 2 \times 0.1 = 0.2 \text{ rad}$$
$$m \theta_2 = 2 \times 0.01 = 0.02 \text{ rad}$$
$$m \theta_3 = 2 \times 0.001 = 0.002 \text{ rad}$$

Computing trig values (to 4 d.p.):

| $i$ | $m\theta_i$ | $\cos(m\theta_i)$ | $\sin(m\theta_i)$ |
|---|---|---|---|
| 0 | 2.000 | $-0.4161$ | $0.9093$ |
| 1 | 0.200 | $0.9801$ | $0.1987$ |
| 2 | 0.020 | $0.9998$ | $0.0200$ |
| 3 | 0.002 | $1.0000$ | $0.0020$ |

The block-diagonal rotation matrix $R_{\Theta, 2} \in \mathbb{R}^{8 \times 8}$ (each $2 \times 2$ block handles one dimension pair):

$$R_{\Theta, 2} = \begin{pmatrix}
-0.4161 & -0.9093 & 0 & 0 & 0 & 0 & 0 & 0 \\
0.9093 & -0.4161 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0.9801 & -0.1987 & 0 & 0 & 0 & 0 \\
0 & 0 & 0.1987 & 0.9801 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0.9998 & -0.0200 & 0 & 0 \\
0 & 0 & 0 & 0 & 0.0200 & 0.9998 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1.0000 & -0.0020 \\
0 & 0 & 0 & 0 & 0 & 0 & 0.0020 & 1.0000
\end{pmatrix}$$

---

### Part C: Proof that $(R_{\Theta,m}\mathbf{q})^T(R_{\Theta,n}\mathbf{k}) = \mathbf{q}^T R_{\Theta, n-m} \mathbf{k}$

**Key property used:** rotation matrices are orthogonal, so $R^T = R^{-1}$ and $R(\alpha)^T R(\beta) = R(\beta - \alpha)$.

**Step 1.** Expand the left side:
$$(R_{\Theta,m}\mathbf{q})^T(R_{\Theta,n}\mathbf{k}) = \mathbf{q}^T R_{\Theta,m}^T R_{\Theta,n}\mathbf{k}$$

**Step 2.** Evaluate $R_{\Theta,m}^T R_{\Theta,n}$ block by block. For the $i$-th diagonal block:

$$R(\theta_i, m)^T R(\theta_i, n)$$

$$= \begin{pmatrix} \cos m\theta_i & \sin m\theta_i \\ -\sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} \cos n\theta_i & -\sin n\theta_i \\ \sin n\theta_i & \cos n\theta_i \end{pmatrix}$$

Multiplying out:

$$= \begin{pmatrix}
\cos m\theta_i \cos n\theta_i + \sin m\theta_i \sin n\theta_i & -\cos m\theta_i \sin n\theta_i + \sin m\theta_i \cos n\theta_i \\
-\sin m\theta_i \cos n\theta_i + \cos m\theta_i \sin n\theta_i & \sin m\theta_i \sin n\theta_i + \cos m\theta_i \cos n\theta_i
\end{pmatrix}$$

Applying the angle difference identities:
- $\cos\alpha\cos\beta + \sin\alpha\sin\beta = \cos(\beta - \alpha)$
- $\sin\alpha\cos\beta - \cos\alpha\sin\beta = \sin(\beta - \alpha)$ (i.e., $\sin(n\theta - m\theta) = \sin(n-m)\theta_i$)

$$= \begin{pmatrix} \cos(n-m)\theta_i & -\sin(n-m)\theta_i \\ \sin(n-m)\theta_i & \cos(n-m)\theta_i \end{pmatrix} = R(\theta_i, n-m)$$

**Step 3.** Since this holds for every block:
$$R_{\Theta,m}^T R_{\Theta,n} = R_{\Theta, n-m}$$

**Conclusion:**
$$(R_{\Theta,m}\mathbf{q})^T(R_{\Theta,n}\mathbf{k}) = \mathbf{q}^T R_{\Theta, n-m} \mathbf{k} \qquad \blacksquare$$

This is the fundamental property that makes RoPE a relative position encoding: the attention logit depends only on the content vectors and the relative offset $n - m$.

---

### Part D: Numerical worked example

**Given:**
- $\mathbf{q} = \mathbf{k} = [1, 0, 0, 1, 1, 0, 0, 1]^T$
- $m = 2$ (query position), $n = 5$ (key position)
- Relative offset: $n - m = 3$

**Step D1: Compute $\tilde{\mathbf{q}} = R_{\Theta, 2} \mathbf{q}$**

The rotation is applied element-wise to adjacent pairs $(q_{2i}, q_{2i+1})$:

**Pair $(q_0, q_1) = (1, 0)$, angle $m\theta_0 = 2.0$ rad:**
$$\tilde{q}_0 = 1 \cdot \cos(2.0) - 0 \cdot \sin(2.0) = -0.4161$$
$$\tilde{q}_1 = 0 \cdot \cos(2.0) + 1 \cdot \sin(2.0) = 0.9093$$

**Pair $(q_2, q_3) = (0, 1)$, angle $m\theta_1 = 0.2$ rad:**
$$\tilde{q}_2 = 0 \cdot \cos(0.2) - 1 \cdot \sin(0.2) = -0.1987$$
$$\tilde{q}_3 = 1 \cdot \cos(0.2) + 0 \cdot \sin(0.2) = 0.9801$$

**Pair $(q_4, q_5) = (1, 0)$, angle $m\theta_2 = 0.02$ rad:**
$$\tilde{q}_4 = 1 \cdot \cos(0.02) - 0 \cdot \sin(0.02) = 0.9998$$
$$\tilde{q}_5 = 0 \cdot \cos(0.02) + 1 \cdot \sin(0.02) = 0.0200$$

**Pair $(q_6, q_7) = (0, 1)$, angle $m\theta_3 = 0.002$ rad:**
$$\tilde{q}_6 = 0 \cdot \cos(0.002) - 1 \cdot \sin(0.002) = -0.0020$$
$$\tilde{q}_7 = 1 \cdot \cos(0.002) + 0 \cdot \sin(0.002) = 1.0000$$

$$\tilde{\mathbf{q}} = [-0.4161,\; 0.9093,\; -0.1987,\; 0.9801,\; 0.9998,\; 0.0200,\; -0.0020,\; 1.0000]^T$$

**Step D2: Compute $\tilde{\mathbf{k}} = R_{\Theta, 5} \mathbf{k}$**

Key angles: $5\theta_0 = 5.0$, $5\theta_1 = 0.5$, $5\theta_2 = 0.05$, $5\theta_3 = 0.005$

| Pair | $(k_{2i}, k_{2i+1})$ | angle | $\tilde{k}_{2i}$ | $\tilde{k}_{2i+1}$ |
|---|---|---|---|---|
| 0 | $(1, 0)$ | 5.000 | $\cos(5) = 0.2837$ | $\sin(5) = -0.9589$ |
| 1 | $(0, 1)$ | 0.500 | $-\sin(0.5) = -0.4794$ | $\cos(0.5) = 0.8776$ |
| 2 | $(1, 0)$ | 0.050 | $\cos(0.05) = 0.9988$ | $\sin(0.05) = 0.0500$ |
| 3 | $(0, 1)$ | 0.005 | $-\sin(0.005) = -0.0050$ | $\cos(0.005) = 1.0000$ |

$$\tilde{\mathbf{k}} = [0.2837,\; -0.9589,\; -0.4794,\; 0.8776,\; 0.9988,\; 0.0500,\; -0.0050,\; 1.0000]^T$$

**Step D3: Compute $\tilde{\mathbf{q}}^T \tilde{\mathbf{k}}$**

$$\tilde{\mathbf{q}}^T \tilde{\mathbf{k}} = (-0.4161)(0.2837) + (0.9093)(-0.9589)$$
$$+ (-0.1987)(-0.4794) + (0.9801)(0.8776)$$
$$+ (0.9998)(0.9988) + (0.0200)(0.0500)$$
$$+ (-0.0020)(-0.0050) + (1.0000)(1.0000)$$

Computing term by term:
- Pair 0: $-0.1181 + (-0.8717) = -0.9898$
- Pair 1: $0.0953 + 0.8601 = 0.9554$
- Pair 2: $0.9986 + 0.0010 = 0.9996$
- Pair 3: $0.0000 + 1.0000 = 1.0000$

$$\tilde{\mathbf{q}}^T \tilde{\mathbf{k}} = -0.9898 + 0.9554 + 0.9996 + 1.0000 = \mathbf{1.9652}$$

**Step D4: Verify via $\mathbf{q}^T R_{\Theta, 3} \mathbf{k}$**

Relative angles: $3\theta_0 = 3.0$, $3\theta_1 = 0.3$, $3\theta_2 = 0.03$, $3\theta_3 = 0.003$

For the same $\mathbf{q} = \mathbf{k} = [1,0,0,1,1,0,0,1]^T$, the product $\mathbf{q}^T R_{\Theta,3} \mathbf{k}$ expands to a sum of $2 \times 2$ block contributions.

For each block pair $(q_{2i}, q_{2i+1})$ and $(k_{2i}, k_{2i+1})$:
$$\text{contribution}_i = q_{2i}[k_{2i}\cos\phi_i - k_{2i+1}\sin\phi_i] + q_{2i+1}[k_{2i+1}\cos\phi_i + k_{2i}\sin\phi_i]$$

where $\phi_i = (n-m)\theta_i = 3\theta_i$.

| $i$ | $(q_{2i},q_{2i+1})$ | $(k_{2i},k_{2i+1})$ | $\phi_i$ | $\cos\phi_i$ | $\sin\phi_i$ | contribution |
|---|---|---|---|---|---|---|
| 0 | $(1,0)$ | $(1,0)$ | 3.000 | $-0.9900$ | $0.1411$ | $-0.9900$ |
| 1 | $(0,1)$ | $(0,1)$ | 0.300 | $0.9553$ | $0.2955$ | $0.9553$ |
| 2 | $(1,0)$ | $(1,0)$ | 0.030 | $0.9996$ | $0.0300$ | $0.9996$ |
| 3 | $(0,1)$ | $(0,1)$ | 0.003 | $1.0000$ | $0.0030$ | $1.0000$ |

$$\mathbf{q}^T R_{\Theta, 3} \mathbf{k} = -0.9900 + 0.9553 + 0.9996 + 1.0000 = 1.9649$$

The small numerical difference from $1.9652$ is rounding error in the trig computations. Analytically the values are identical, confirming the proof. $\checkmark$

---

### Part E: Intuitive interpretation — attention logit vs distance

The attention logit $\mathbf{q}^T R_{\Theta, n-m} \mathbf{k}$ can be written block-wise:

$$\sum_{i=0}^{d/2-1} \left[ q_{2i} k_{2i} \cos((n-m)\theta_i) + q_{2i+1} k_{2i+1} \cos((n-m)\theta_i) + \text{cross terms} \right]$$

For a "generic" pair of vectors, the cosine terms oscillate as $n - m$ increases. For large $\theta_i$ (high frequency, dimension 0), the cosine oscillates rapidly — the contribution from this dimension averages to approximately 0 over many different distances. For small $\theta_i$ (low frequency, dimension 3), the cosine stays near 1 for large distances — this dimension contributes a nearly constant value for any relative position.

**Net effect:** As $|n-m|$ increases:
- High-frequency dimensions decorrelate quickly (cosine oscillates)
- Low-frequency dimensions remain correlated longer
- The expected magnitude $\mathbb{E}[|q^T R_{\Theta,\delta} k|]$ decreases with $|\delta|$, implementing a soft distance bias

This is an implicit distance bias similar to ALiBi's explicit linear penalty, but more expressive because it is content-dependent (the actual query and key values interact with the rotation).

**Comparison for this example:** At $\delta = 3$, the high-frequency pair (i=0) contributed $-0.99$ while at $\delta = 0$ it would contribute $+1.0$ — the sign has flipped. The low-frequency pairs (i=2,3) contributed $\approx +1.0$ in both cases, as expected for slow-varying dimensions.

---

## Implementation Reference

```python
import torch
import math

def compute_rope_frequencies(d_k: int, base: float = 10000.0) -> torch.Tensor:
    """Compute RoPE frequencies theta_i for each dimension pair."""
    # Shape: (d_k // 2,)
    i = torch.arange(0, d_k, 2, dtype=torch.float32)
    return 1.0 / (base ** (i / d_k))  # theta_i = base^{-2i/d_k}

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply RoPE to query or key tensor.

    Args:
        x: shape (batch, seq_len, n_heads, d_k)
        positions: shape (seq_len,) -- integer position indices
        base: RoPE base frequency

    Returns:
        Rotated tensor of same shape as x.
    """
    batch, seq_len, n_heads, d_k = x.shape
    assert d_k % 2 == 0

    # Frequencies: (d_k//2,)
    freqs = compute_rope_frequencies(d_k, base)

    # Angles: (seq_len, d_k//2)
    angles = positions.unsqueeze(1).float() * freqs.unsqueeze(0)  # outer product

    # cos and sin: (seq_len, d_k)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # duplicate each for pairs
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)

    # Broadcast to (batch, seq_len, n_heads, d_k)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Create x_rot: the "rotated partner" of each adjacent pair
    # For pair (x[2i], x[2i+1]) -> x_rot = (-x[2i+1], x[2i])
    x_rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)

    return x * cos + x_rot * sin

# Verify Part D numerically
d_k = 8
q = torch.tensor([[[[1., 0., 0., 1., 1., 0., 0., 1.]]]])  # (1, 1, 1, 8)
k = torch.tensor([[[[1., 0., 0., 1., 1., 0., 0., 1.]]]])  # (1, 1, 1, 8)

pos_q = torch.tensor([2])  # position m=2
pos_k = torch.tensor([5])  # position n=5

q_rot = apply_rope(q, pos_q)
k_rot = apply_rope(k, pos_k)

logit_direct = (q_rot * k_rot).sum()
print(f"Direct logit (m=2, n=5): {logit_direct.item():.4f}")
# Expected: ~1.9652

# Verify relative encoding: compute at relative offset 3
q_at_0 = apply_rope(q, torch.tensor([0]))
k_at_3 = apply_rope(k, torch.tensor([3]))
logit_relative = (q_at_0 * k_at_3).sum()
print(f"Relative logit (offset=3): {logit_relative.item():.4f}")
# Should match logit_direct
```

**Expected output:**
```
Direct logit (m=2, n=5): 1.9649
Relative logit (offset=3): 1.9649
```

Both values match, confirming that RoPE produces position-independent attention scores that depend only on relative offset.
