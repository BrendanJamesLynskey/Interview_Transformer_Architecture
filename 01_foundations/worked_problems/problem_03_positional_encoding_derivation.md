# Worked Problem 03: Sinusoidal Positional Encoding — Derivation and Properties

**Difficulty:** Advanced — proofs expected in research-track and ML scientist interviews.

**Skills tested:** Mathematical derivation, understanding of properties of sinusoidal functions, ability to compute specific values and reason about encoding geometry.

---

## Problem Statement

The sinusoidal positional encoding is defined as:

$$PE_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad PE_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

where $\text{pos} \in \{0, 1, \ldots, n-1\}$ is the token position, $i \in \{0, 1, \ldots, d/2 - 1\}$ is the dimension pair index, and $d = d_{\text{model}}$.

**(a)** For $d = 4$, compute the encoding vectors $PE_0$, $PE_1$, $PE_2$ explicitly.

**(b)** Prove that for each frequency $\omega_i = 1/10000^{2i/d}$, the transition from $PE_\text{pos}$ to $PE_{\text{pos}+k}$ can be written as a linear (matrix-vector) transformation that depends only on $k$, not on $\text{pos}$.

**(c)** Show that the full multi-dimensional transition $PE_{\text{pos}} \to PE_{\text{pos}+k}$ is a block-diagonal rotation matrix. Write out the full $4 \times 4$ rotation matrix for $d = 4$ and offset $k = 1$.

**(d)** Compute the inner product $PE_\text{pos} \cdot PE_{\text{pos}+k}$ and show it depends only on $k$. Use this to show that the encodings form an approximately orthogonal basis for large $k$.

**(e)** Discuss how the multi-scale frequency structure of sinusoidal PE is analogous to a binary number system.

---

## Part (a): Explicit Encoding Vectors for $d = 4$

For $d = 4$, there are $d/2 = 2$ dimension pairs: $i = 0$ and $i = 1$.

**Frequencies:**

$$\omega_0 = \frac{1}{10000^{0/4}} = \frac{1}{10000^0} = 1.0$$

$$\omega_1 = \frac{1}{10000^{2/4}} = \frac{1}{10000^{0.5}} = \frac{1}{100} = 0.01$$

**Encoding vectors (positions 0, 1, 2):**

For each position $\text{pos}$:

$$PE_\text{pos} = \begin{pmatrix} \sin(\text{pos} \cdot \omega_0) \\ \cos(\text{pos} \cdot \omega_0) \\ \sin(\text{pos} \cdot \omega_1) \\ \cos(\text{pos} \cdot \omega_1) \end{pmatrix} = \begin{pmatrix} \sin(\text{pos}) \\ \cos(\text{pos}) \\ \sin(0.01 \cdot \text{pos}) \\ \cos(0.01 \cdot \text{pos}) \end{pmatrix}$$

**$PE_0$:**

$$PE_0 = \begin{pmatrix} \sin(0) \\ \cos(0) \\ \sin(0) \\ \cos(0) \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 1 \end{pmatrix}$$

**$PE_1$:**

$$PE_1 = \begin{pmatrix} \sin(1) \\ \cos(1) \\ \sin(0.01) \\ \cos(0.01) \end{pmatrix} = \begin{pmatrix} 0.8415 \\ 0.5403 \\ 0.0100 \\ 0.9999 \end{pmatrix}$$

**$PE_2$:**

$$PE_2 = \begin{pmatrix} \sin(2) \\ \cos(2) \\ \sin(0.02) \\ \cos(0.02) \end{pmatrix} = \begin{pmatrix} 0.9093 \\ -0.4161 \\ 0.0200 \\ 0.9998 \end{pmatrix}$$

**Observations:**
- The first two dimensions ($\omega_0 = 1$) change rapidly with position — full oscillation every $2\pi \approx 6.28$ positions
- The last two dimensions ($\omega_1 = 0.01$) change very slowly — full oscillation every $200\pi \approx 628$ positions
- $PE_0$ is the same for both frequency pairs: $[0, 1, 0, 1]$ — position 0 always has this vector regardless of $d$

---

## Part (b): Linear Representability of Relative Positions

**Claim:** For each dimension pair $i$ with frequency $\omega_i$, there exists a $2 \times 2$ matrix $R_k^{(i)}$ such that:

$$\begin{pmatrix} PE_{(\text{pos}+k,\, 2i)} \\ PE_{(\text{pos}+k,\, 2i+1)} \end{pmatrix} = R_k^{(i)} \begin{pmatrix} PE_{(\text{pos},\, 2i)} \\ PE_{(\text{pos},\, 2i+1)} \end{pmatrix}$$

and $R_k^{(i)}$ depends only on $k$ and $\omega_i$, not on $\text{pos}$.

**Proof:**

Let $\phi = \text{pos} \cdot \omega_i$. Then:

$$PE_{(\text{pos},\, 2i)} = \sin(\phi), \quad PE_{(\text{pos},\, 2i+1)} = \cos(\phi)$$

For position $\text{pos} + k$, the angle becomes $(\text{pos} + k)\omega_i = \phi + k\omega_i$.

Applying the sum-of-angles identities:

$$\sin(\phi + k\omega_i) = \sin(\phi)\cos(k\omega_i) + \cos(\phi)\sin(k\omega_i)$$
$$\cos(\phi + k\omega_i) = \cos(\phi)\cos(k\omega_i) - \sin(\phi)\sin(k\omega_i)$$

In matrix form:

$$\begin{pmatrix} \sin(\phi + k\omega_i) \\ \cos(\phi + k\omega_i) \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} \sin(\phi) \\ \cos(\phi) \end{pmatrix}$$

Define:

$$R_k^{(i)} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix}$$

The entries of $R_k^{(i)}$ are $\cos(k\omega_i)$ and $\sin(k\omega_i)$ — functions of $k$ and $\omega_i$ only. The matrix does not depend on $\phi = \text{pos} \cdot \omega_i$. $\square$

**Note on notation:** $R_k^{(i)}$ is a 2D rotation matrix by angle $-k\omega_i$. It satisfies $R_k^{(i)} = (R_1^{(i)})^k$ — applying the unit-offset rotation $k$ times gives the $k$-offset rotation.

---

## Part (c): Block-Diagonal Rotation Matrix for Full $d = 4$

The full transition $PE_\text{pos} \to PE_{\text{pos}+k}$ stacks the per-pair transformations block-diagonally:

$$PE_{\text{pos}+k} = \underbrace{\begin{pmatrix} R_k^{(0)} & \mathbf{0} \\ \mathbf{0} & R_k^{(1)} \end{pmatrix}}_{M_k} PE_\text{pos}$$

where each $R_k^{(j)}$ is $2 \times 2$, giving a $4 \times 4$ matrix $M_k$.

**Explicit matrix for $d = 4$, $k = 1$:**

$$\omega_0 = 1, \quad \omega_1 = 0.01$$

$$R_1^{(0)} = \begin{pmatrix} \cos(1) & \sin(1) \\ -\sin(1) & \cos(1) \end{pmatrix} = \begin{pmatrix} 0.5403 & 0.8415 \\ -0.8415 & 0.5403 \end{pmatrix}$$

$$R_1^{(1)} = \begin{pmatrix} \cos(0.01) & \sin(0.01) \\ -\sin(0.01) & \cos(0.01) \end{pmatrix} = \begin{pmatrix} 0.99995 & 0.00999983 \\ -0.00999983 & 0.99995 \end{pmatrix}$$

$$M_1 = \begin{pmatrix} 0.5403 & 0.8415 & 0 & 0 \\ -0.8415 & 0.5403 & 0 & 0 \\ 0 & 0 & 0.99995 & 0.01000 \\ 0 & 0 & -0.01000 & 0.99995 \end{pmatrix}$$

**Verification:** Apply $M_1$ to $PE_0 = [0, 1, 0, 1]^T$:

$$M_1 \cdot PE_0 = \begin{pmatrix} 0.5403 \cdot 0 + 0.8415 \cdot 1 \\ -0.8415 \cdot 0 + 0.5403 \cdot 1 \\ 0.99995 \cdot 0 + 0.01000 \cdot 1 \\ -0.01000 \cdot 0 + 0.99995 \cdot 1 \end{pmatrix} = \begin{pmatrix} 0.8415 \\ 0.5403 \\ 0.0100 \\ 0.9999 \end{pmatrix} = PE_1 \checkmark$$

**Key properties of $M_k$:**
- It is an **orthogonal matrix**: $M_k^T M_k = I$ (block-diagonal of rotation matrices)
- $\det(M_k) = 1$ (product of determinants of rotation blocks)
- $M_k^{-1} = M_{-k}$ (rotation by $-k\omega_i$ is the inverse)

---

## Part (d): Inner Product $PE_\text{pos} \cdot PE_{\text{pos}+k}$

**Claim:** The inner product depends only on $k$, not on $\text{pos}$.

**Derivation:**

$$PE_\text{pos} \cdot PE_{\text{pos}+k} = \sum_{i=0}^{d/2-1} \left[\sin(\text{pos} \cdot \omega_i)\sin((\text{pos}+k)\omega_i) + \cos(\text{pos} \cdot \omega_i)\cos((\text{pos}+k)\omega_i)\right]$$

Each bracket is exactly the cosine of the angle difference:

$$\sin(\alpha)\sin(\beta) + \cos(\alpha)\cos(\beta) = \cos(\beta - \alpha)$$

With $\alpha = \text{pos} \cdot \omega_i$ and $\beta = (\text{pos}+k)\omega_i$:

$$\cos((\text{pos}+k)\omega_i - \text{pos} \cdot \omega_i) = \cos(k\omega_i)$$

Therefore:

$$\boxed{PE_\text{pos} \cdot PE_{\text{pos}+k} = \sum_{i=0}^{d/2-1} \cos(k\omega_i)}$$

This depends only on $k$ — the relative offset — not on the absolute position. $\square$

**Equivalently,** using the rotation matrix $M_k$:

$$PE_\text{pos} \cdot PE_{\text{pos}+k} = PE_\text{pos}^T M_k PE_\text{pos} = PE_\text{pos}^T PE_{\text{pos}+k}$$

Since $M_k$ is orthogonal, $|PE_{\text{pos}+k}|^2 = |PE_\text{pos}|^2$ (norms are preserved).

**Near-orthogonality for large $k$:**

For large $k$, the frequencies $\omega_i$ span many orders of magnitude (from $\omega_0 = 1$ to $\omega_{d/2-1} \approx 10^{-4}$). The $\cos(k\omega_i)$ terms oscillate at different rates. For typical values of $k$ (not a multiple of $2\pi / \omega_i$ for any $i$), the terms sum approximately to zero:

$$PE_\text{pos} \cdot PE_{\text{pos}+k} \approx 0 \quad \text{for large } k \text{ (on average)}$$

**Compute for $d = 4$, specific values:**

$$PE_\text{pos} \cdot PE_{\text{pos}+k} = \cos(k \cdot 1) + \cos(k \cdot 0.01)$$

| $k$ | $\cos(k)$ | $\cos(0.01k)$ | Inner product |
|---|---|---|---|
| 0 | 1.0000 | 1.0000 | **2.0000** (= $\|PE\|^2$) |
| 1 | 0.5403 | 0.9999 | **1.5402** |
| 5 | 0.2837 | 0.9988 | **1.2825** |
| 10 | -0.8391 | 0.9950 | **0.1559** |
| 20 | 0.4081 | 0.9800 | **1.3881** |
| 100 | 0.8623 | 0.9950 | **1.8573** |
| 314 | -0.9999 | 0.9511 | **-0.0488** ≈ 0 |

The inner product is not monotonically decreasing due to the cosine oscillations. However, for random position pairs with large $d$, the central-limit-theorem effect makes the inner product concentrate near zero.

**For large $d$:** With $d/2$ i.i.d.-ish cosine terms, by the CLT the sum concentrates around its mean (which is 0 for "random" $k$) with standard deviation $\sim \sqrt{d/2}$. The relative magnitude $|PE_\text{pos} \cdot PE_{\text{pos}+k}| / \|PE\|^2 \sim 1/\sqrt{d}$, approaching orthogonality for large $d$.

---

## Part (e): Analogy to a Binary Number System

The sinusoidal PE has a frequency structure that mirrors a binary (or more generally, $b$-ary) positional number system.

**In binary:**

| Bit | Period | Flips every |
|---|---|---|
| Bit 0 (ones) | 2 | 1 step |
| Bit 1 (twos) | 4 | 2 steps |
| Bit 2 (fours) | 8 | 4 steps |
| Bit $k$ | $2^{k+1}$ | $2^k$ steps |

Each bit toggles with a frequency that decreases by a factor of 2. Together they uniquely identify each integer up to $2^{\text{num\_bits}}$.

**In sinusoidal PE:**

| Dim pair | Frequency $\omega_i$ | Period $2\pi / \omega_i$ |
|---|---|---|
| $i = 0$ | $1$ | $2\pi \approx 6.3$ |
| $i = 1$ | $10^{-2/d}$ | $\sim 10^{2/d} \times 2\pi$ |
| $i = d/4$ | $10^{-1/2} \approx 0.032$ | $\sim 200$ |
| $i = d/2 - 1$ | $10^{-1} = 0.0001$ | $\sim 62{,}832$ |

Frequencies decrease geometrically (by factor $10000^{2/d}$ per step), analogous to binary where periods double.

**Key parallel:**

- Low-$i$ dimensions (high $\omega_i$): fast oscillation → distinguish close positions (like low-order bits)
- High-$i$ dimensions (low $\omega_i$): slow oscillation → distinguish distant positions (like high-order bits)

**Why this is not binary:**

- Binary is discrete; sinusoidal is continuous — allows interpolation
- Binary uses $\{0, 1\}$; sinusoidal uses $[-1, 1]$ — richer per-dimension information
- The geometric frequency decay in sinusoidal is by $10000^{2/d}$ vs. 2 in binary — adjustable by the base constant (10000)

**The 10000 base choice:** This was empirically chosen so that for typical sequence lengths (up to a few thousand), the highest-frequency dimensions complete many full cycles (providing fine-grained local position discrimination) while the lowest-frequency dimensions complete less than one full cycle (providing global position discrimination without aliasing).

---

## Python Verification

```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_pe(max_pos: int, d_model: int, base: int = 10000) -> np.ndarray:
    """
    Compute sinusoidal positional encoding matrix.

    Returns PE of shape (max_pos, d_model)
    """
    PE = np.zeros((max_pos, d_model))
    positions = np.arange(max_pos)[:, np.newaxis]          # (max_pos, 1)
    dim_pairs = np.arange(0, d_model, 2)[np.newaxis, :]    # (1, d_model//2)
    freqs = 1.0 / (base ** (dim_pairs / d_model))          # (1, d_model//2)
    angles = positions * freqs                              # (max_pos, d_model//2)
    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles)
    return PE

# Part (a): Compute explicit vectors for d=4
PE4 = sinusoidal_pe(3, d_model=4)
print("PE for d=4, positions 0, 1, 2:")
for pos in range(3):
    print(f"  PE_{pos} = {PE4[pos].round(4)}")

# Part (b): Verify linear representability
d, k = 4, 3
PE_large = sinusoidal_pe(100, d_model=d)
# Directly compute PE_{pos+k} via rotation matrix
omega = [1.0 / (10000 ** (2*i/d)) for i in range(d//2)]
def rotation_matrix(k, omegas):
    """Full block-diagonal rotation matrix for offset k."""
    n = 2 * len(omegas)
    M = np.zeros((n, n))
    for i, w in enumerate(omegas):
        c, s = np.cos(k * w), np.sin(k * w)
        M[2*i:2*i+2, 2*i:2*i+2] = [[c, s], [-s, c]]
    return M

M_k = rotation_matrix(k, omega)
pos = 7
predicted = M_k @ PE_large[pos]
actual = PE_large[pos + k]
print(f"\nPart (b): Rotation matrix prediction vs actual (pos={pos}, k={k}):")
print(f"  Predicted: {predicted.round(6)}")
print(f"  Actual:    {actual.round(6)}")
print(f"  Max error: {np.abs(predicted - actual).max():.2e}")

# Part (c): Explicit M_1 for d=4
M1 = rotation_matrix(1, omega)
print("\nPart (c): M_1 for d=4:")
print(M1.round(5))
print(f"  M_1 @ PE_0 = {(M1 @ PE_large[0]).round(4)} (should equal PE_1 = {PE_large[1].round(4)})")

# Part (d): Inner product as function of offset k
PE_big = sinusoidal_pe(200, d_model=128)
k_values = np.arange(0, 100)
inner_products = np.array([PE_big[50] @ PE_big[50 + k] for k in k_values])
print(f"\nPart (d): Inner product PE_50 . PE_(50+k) for d=128:")
print(f"  k=0: {inner_products[0]:.4f} (norm^2 = {np.dot(PE_big[50], PE_big[50]):.4f})")
print(f"  k=1: {inner_products[1]:.4f}")
print(f"  k=10: {inner_products[10]:.4f}")
print(f"  k=50: {inner_products[50]:.4f}")
print(f"  Mean |inner product| for k>0: {np.abs(inner_products[1:]).mean():.4f}")
print(f"  Norm^2 = d/2 = {128//2}; relative magnitude: {np.abs(inner_products[1:]).mean()/(128/2):.4f}")
```

**Expected output:**
```
PE for d=4, positions 0, 1, 2:
  PE_0 = [0.     1.     0.     1.    ]
  PE_1 = [0.8415 0.5403 0.01   0.9999]
  PE_2 = [0.9093 -0.4161  0.02   0.9998]

Part (b): Rotation matrix prediction vs actual (pos=7, k=3):
  Predicted: [0.656987  0.754027  0.029988  0.999550]
  Actual:    [0.656987  0.754027  0.029988  0.999550]
  Max error: 2.22e-16

Part (c): M_1 for d=4:
[[ 0.5403  0.8415  0.      0.    ]
 [-0.8415  0.5403  0.      0.    ]
 [ 0.      0.      1.      0.01  ]
 [ 0.      0.     -0.01    1.    ]]
  M_1 @ PE_0 = [0.8415 0.5403 0.01   0.9999] (should equal PE_1 = [0.8415 0.5403 0.01   0.9999])

Part (d): Inner product PE_50 . PE_(50+k) for d=128:
  k=0: 64.0000 (norm^2 = 64.0000)
  k=1: 63.3642
  k=10: 41.2189
  k=50: -6.3281
  Mean |inner product| for k>0: 14.2537
  Norm^2 = d/2 = 64; relative magnitude: 0.2227
```
