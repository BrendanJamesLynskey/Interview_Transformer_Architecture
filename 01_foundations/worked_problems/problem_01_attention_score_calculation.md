# Worked Problem 01: Attention Score Calculation

**Difficulty:** Fundamentals — suitable as a whiteboard problem in ML engineering interviews.

**Skills tested:** Understanding of scaled dot-product attention, ability to execute matrix operations, numerical intuition about softmax.

---

## Problem Statement

You are given the following matrices representing queries, keys, and values for a single-head attention layer with $d_k = d_v = 4$ and sequence length $n = 3$.

$$Q = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{pmatrix}, \quad V = \begin{pmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}$$

Note: $V \in \mathbb{R}^{3 \times 2}$, so $d_v = 2$ (for computational tractability while keeping $d_k = 4$).

**(a)** Compute the raw (unscaled) attention score matrix $QK^T$.

**(b)** Apply the scaling factor $1/\sqrt{d_k}$ to obtain the scaled scores.

**(c)** Apply the softmax function row-wise to obtain the attention weight matrix $A$.

**(d)** Compute the output $O = AV$.

**(e)** Now apply a causal mask before the softmax and recompute the output. Which tokens' outputs change and why?

---

## Full Solution

### Part (a): Compute $QK^T$

$Q \in \mathbb{R}^{3 \times 4}$, $K \in \mathbb{R}^{3 \times 4}$, so $K^T \in \mathbb{R}^{4 \times 3}$.

$$K^T = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{pmatrix}$$

$$QK^T = \begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{pmatrix}$$

**Row 1** $(q_1 = [1, 0, 1, 0])$:
- $q_1 \cdot k_1 = 1 \cdot 1 + 0 \cdot 0 + 1 \cdot 0 + 0 \cdot 1 = 1$
- $q_1 \cdot k_2 = 1 \cdot 0 + 0 \cdot 1 + 1 \cdot 1 + 0 \cdot 0 = 1$
- $q_1 \cdot k_3 = 1 \cdot 1 + 0 \cdot 1 + 1 \cdot 1 + 0 \cdot 1 = 2$

**Row 2** $(q_2 = [0, 1, 0, 1])$:
- $q_2 \cdot k_1 = 0 + 0 + 0 + 1 = 1$
- $q_2 \cdot k_2 = 0 + 1 + 0 + 0 = 1$
- $q_2 \cdot k_3 = 0 + 1 + 0 + 1 = 2$

**Row 3** $(q_3 = [1, 1, 0, 0])$:
- $q_3 \cdot k_1 = 1 + 0 + 0 + 0 = 1$
- $q_3 \cdot k_2 = 0 + 1 + 0 + 0 = 1$
- $q_3 \cdot k_3 = 1 + 1 + 0 + 0 = 2$

$$QK^T = \begin{pmatrix} 1 & 1 & 2 \\ 1 & 1 & 2 \\ 1 & 1 & 2 \end{pmatrix}$$

**Observation:** All three rows are identical. This means every query, despite being different, assigns the same relative attention to the keys. This is a property of this specific example — in practice, different queries produce different score distributions.

---

### Part (b): Apply scaling $1/\sqrt{d_k}$

$d_k = 4$, so $\sqrt{d_k} = 2$.

$$\frac{QK^T}{\sqrt{d_k}} = \frac{1}{2} \begin{pmatrix} 1 & 1 & 2 \\ 1 & 1 & 2 \\ 1 & 1 & 2 \end{pmatrix} = \begin{pmatrix} 0.5 & 0.5 & 1.0 \\ 0.5 & 0.5 & 1.0 \\ 0.5 & 0.5 & 1.0 \end{pmatrix}$$

---

### Part (c): Apply softmax row-wise

For row $[0.5, 0.5, 1.0]$, the softmax computation:

**Step 1: Subtract row max for numerical stability**
$$\max = 1.0, \quad [0.5 - 1.0,\ 0.5 - 1.0,\ 1.0 - 1.0] = [-0.5,\ -0.5,\ 0.0]$$

**Step 2: Exponentiate**
$$[e^{-0.5},\ e^{-0.5},\ e^{0}] = [0.6065,\ 0.6065,\ 1.0000]$$

**Step 3: Normalise**
$$\text{sum} = 0.6065 + 0.6065 + 1.0000 = 2.2130$$

$$\alpha_1 = \alpha_2 = \frac{0.6065}{2.2130} \approx 0.2742, \quad \alpha_3 = \frac{1.0000}{2.2130} \approx 0.4517$$

**Verification:** $0.2742 + 0.2742 + 0.4517 = 1.0001 \approx 1.0$ ✓

Since all rows are identical, the attention weight matrix is:

$$A = \begin{pmatrix} 0.2742 & 0.2742 & 0.4517 \\ 0.2742 & 0.2742 & 0.4517 \\ 0.2742 & 0.2742 & 0.4517 \end{pmatrix}$$

**Interpretation:** Every token query assigns approximately 27% of its attention to tokens 1 and 2, and 45% to token 3. Token 3 (whose key vector $[1,1,1,1]$ has larger dot products with all queries) is the most attended-to.

---

### Part (d): Compute output $O = AV$

$$V = \begin{pmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}$$

For any row $i$ of $A$ (they are all identical), the output row is:

$$o_i = \alpha_1 v_1 + \alpha_2 v_2 + \alpha_3 v_3$$

$$o_i = 0.2742 \begin{pmatrix}2\\0\end{pmatrix} + 0.2742 \begin{pmatrix}0\\2\end{pmatrix} + 0.4517 \begin{pmatrix}1\\1\end{pmatrix}$$

**Dimension 1:** $0.2742 \times 2 + 0.2742 \times 0 + 0.4517 \times 1 = 0.5484 + 0 + 0.4517 = 1.0001 \approx 1.0$

**Dimension 2:** $0.2742 \times 0 + 0.2742 \times 2 + 0.4517 \times 1 = 0 + 0.5484 + 0.4517 = 1.0001 \approx 1.0$

$$O = \begin{pmatrix} 1.0 & 1.0 \\ 1.0 & 1.0 \\ 1.0 & 1.0 \end{pmatrix}$$

**Interpretation:** All three output tokens are identical — a consequence of all three rows of $A$ being the same. The attention-weighted average of the value vectors collapses to the same vector for all positions.

---

### Part (e): Apply causal mask and recompute

The causal mask sets $-\infty$ for all positions where $j > i$ (future tokens):

$$M = \begin{pmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{pmatrix}$$

**Masked scaled scores:**

$$\frac{QK^T}{\sqrt{d_k}} + M = \begin{pmatrix} 0.5 & -\infty & -\infty \\ 0.5 & 0.5 & -\infty \\ 0.5 & 0.5 & 1.0 \end{pmatrix}$$

**Row-wise softmax:**

**Row 1** $[0.5, -\infty, -\infty]$: Only position 1 is valid.

$$A_1 = [1.0,\ 0.0,\ 0.0]$$

**Row 2** $[0.5, 0.5, -\infty]$: Two equal valid scores.

$$e^{0.5} = 1.6487, \quad \text{sum} = 2 \times 1.6487 = 3.2974$$

$$A_2 = [0.5,\ 0.5,\ 0.0]$$

**Row 3** $[0.5, 0.5, 1.0]$: Same as before (all three valid).

$$A_3 = [0.2742,\ 0.2742,\ 0.4517]$$

**Masked attention matrix:**

$$A^{\text{causal}} = \begin{pmatrix} 1.0 & 0.0 & 0.0 \\ 0.5 & 0.5 & 0.0 \\ 0.2742 & 0.2742 & 0.4517 \end{pmatrix}$$

**Compute masked output $O^{\text{causal}} = A^{\text{causal}} V$:**

**Row 1:** $1.0 \times [2, 0] = [2.0, 0.0]$

**Row 2:** $0.5 \times [2, 0] + 0.5 \times [0, 2] = [1.0, 0.0] + [0.0, 1.0] = [1.0, 1.0]$

**Row 3:** (unchanged from part d) $= [1.0, 1.0]$

$$O^{\text{causal}} = \begin{pmatrix} 2.0 & 0.0 \\ 1.0 & 1.0 \\ 1.0 & 1.0 \end{pmatrix}$$

**Which tokens' outputs changed?**

- **Token 1:** Changed from $[1.0, 1.0]$ to $[2.0, 0.0]$. Now attends only to itself (value $v_1 = [2, 0]$).
- **Token 2:** Changed from $[1.0, 1.0]$ to $[1.0, 1.0]$. By coincidence, the average of $v_1$ and $v_2$ equals the causal average.
- **Token 3:** Unchanged — all three positions were available to it anyway.

**Key takeaway:** The causal mask has the greatest impact on early tokens in the sequence. The first token is most severely constrained (can only attend to itself). Later tokens progressively have more context available, until the last token in training has access to the full sequence. At test time during generation, token $t$ always has exactly $t$ tokens of context.

---

## Python Verification

```python
import numpy as np

# Define matrices
Q = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 1, 0, 0]], dtype=float)

K = np.array([[1, 0, 0, 1],
              [0, 1, 1, 0],
              [1, 1, 1, 1]], dtype=float)

V = np.array([[2, 0],
              [0, 2],
              [1, 1]], dtype=float)

d_k = Q.shape[-1]  # 4

# (a) Raw scores
raw_scores = Q @ K.T
print("QK^T:\n", raw_scores)

# (b) Scaled scores
scaled_scores = raw_scores / np.sqrt(d_k)
print("\nScaled scores:\n", scaled_scores)

# (c) Softmax
def softmax_rows(x):
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

A = softmax_rows(scaled_scores)
print("\nAttention weights A:\n", np.round(A, 4))
print("Row sums:", A.sum(axis=-1))

# (d) Output
O = A @ V
print("\nOutput O:\n", np.round(O, 4))

# (e) Causal mask
n = Q.shape[0]
causal_mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # True = mask out
masked_scores = scaled_scores.copy()
masked_scores[causal_mask] = -np.inf

A_causal = softmax_rows(masked_scores)
print("\nCausal attention weights:\n", np.round(A_causal, 4))

O_causal = A_causal @ V
print("\nCausal output:\n", np.round(O_causal, 4))
```

**Output:**
```
QK^T:
 [[1. 1. 2.]
  [1. 1. 2.]
  [1. 1. 2.]]

Scaled scores:
 [[0.5 0.5 1. ]
  [0.5 0.5 1. ]
  [0.5 0.5 1. ]]

Attention weights A:
 [[0.2742 0.2742 0.4517]
  [0.2742 0.2742 0.4517]
  [0.2742 0.2742 0.4517]]
Row sums: [1. 1. 1.]

Output O:
 [[1. 1.]
  [1. 1.]
  [1. 1.]]

Causal attention weights:
 [[1.     0.     0.    ]
  [0.5    0.5    0.    ]
  [0.2742 0.2742 0.4517]]

Causal output:
 [[2. 0.]
  [1. 1.]
  [1. 1.]]
```

---

## Common Mistakes to Avoid

1. **Forgetting to transpose $K$.** The formula is $QK^T$, not $QK$. Shapes: $(n \times d_k)(d_k \times n) \to (n \times n)$.

2. **Scaling by $d_k$ instead of $\sqrt{d_k}$.** The variance argument requires dividing by the standard deviation $\sqrt{d_k}$, not the variance $d_k$.

3. **Applying softmax column-wise instead of row-wise.** Each row corresponds to one query's distribution over all keys. Columns do not sum to 1.

4. **Not applying the causal mask before softmax.** If you set future weights to zero after softmax and renormalise, the result differs — you must use $-\infty$ before softmax to ensure the normalisation is correct.

5. **Ignoring numerical stability.** For exam problems, exact computation is fine. In implementation, always subtract the row maximum before exponentiating.
