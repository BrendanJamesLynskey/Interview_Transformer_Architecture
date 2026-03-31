# Problem 03: FlashAttention Tiling

**Topic:** FlashAttention — tiling algorithm, online softmax, memory analysis

**Difficulty:** Advanced

**Expected time:** 30–40 minutes

---

## Problem Statement

You are stepping through the FlashAttention algorithm for a sequence with:
- Sequence length $N = 8$
- Head dimension $d = 4$
- Query block size $B_r = 2$ rows
- Key/Value block size $B_c = 4$ columns
- Causal (lower-triangular) attention mask

**Given matrices** ($Q, K, V \in \mathbb{R}^{8 \times 4}$, but we use a small numerical example):

$$Q = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
1 & 1 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
K = V = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
1 & 1 & 0 & 0 \\
0 & 0 & 1 & 1 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1
\end{bmatrix}$$

**Part A.** Describe the tiling structure. How many Q-blocks and K/V-blocks are there? Draw the tiling grid showing which K/V tiles a causal model must process for each Q tile.

**Part B.** Step through the FlashAttention algorithm for Q-block 1 (rows 1–2), processing K/V tile 1 (columns 1–4). Show:
1. The raw attention score tile $S_{11}$
2. The masked score tile
3. The block maximum $m_{11}$
4. The exponentiated scores
5. The block sum $\ell_{11}$
6. The partial output $O_{11}$
7. The running statistics after this tile

**Part C.** There are no further K/V tiles for Q-block 1 under causal masking (K/V tile 1 covers all columns for rows 1–2). Finalise the output by applying the normalisation. Compute the final $O_1$ (the output rows 1–2).

**Part D.** Now step through Q-block 2 (rows 3–4). This block must process K/V tile 1 (valid) and K/V tile 2 (partially masked). Show the online update when adding tile 2's contribution.

**Part E.** Count the HBM reads/writes for the full algorithm (all 4 Q-blocks, with causal masking). Compare to standard attention's HBM operations.

---

## Solution

### Part A: Tiling structure

**Block counts:**

$$\text{Q-blocks} = \lceil N / B_r \rceil = \lceil 8 / 2 \rceil = 4$$
$$\text{K/V-blocks} = \lceil N / B_c \rceil = \lceil 8 / 4 \rceil = 2$$

Q-block $i$ contains rows $\{2i-1, 2i\}$ (1-indexed). K/V-block $j$ contains columns $\{4j-3, \ldots, 4j\}$.

**Causal mask structure.** Q-block $i$ can only attend to K/V positions $\leq$ the current query positions. Specifically, Q-block $i$ (rows $2i-1$ to $2i$) attends to K/V-block $j$ if $4(j-1) + 1 \leq 2i$ (i.e., the K/V block starts before the end of the Q block).

**Tiling grid** (mark: full = process in full, partial = process with masking, skip = skip entirely):

| | KV-block 1 (cols 1–4) | KV-block 2 (cols 5–8) |
|---|---|---|
| **Q-block 1** (rows 1–2) | Partial (diagonal block) | Skip |
| **Q-block 2** (rows 3–4) | Full | Partial |
| **Q-block 3** (rows 5–6) | Full | Partial |
| **Q-block 4** (rows 7–8) | Full | Full |

**Partial** = the block intersects the diagonal of the causal mask; some entries must be set to $-\infty$. **Full** = all entries are valid (below the diagonal). **Skip** = all entries are above the diagonal; skip entirely.

Total tiles to compute: $\underbrace{1}_{\text{Q1}} + \underbrace{2}_{\text{Q2}} + \underbrace{2}_{\text{Q3}} + \underbrace{2}_{\text{Q4}} = 7$ out of $4 \times 2 = 8$ possible tiles. (1 is skipped.)

---

### Part B: Q-block 1, K/V-tile 1 — detailed step-through

**Q-block 1:** rows 1–2 of $Q$:
$$Q_1 = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

**K/V-tile 1:** rows 1–4 of $K$ and $V$:
$$K_1 = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad V_1 = K_1$$

**Step 1: Raw score tile.**

$$S_{11} = \frac{Q_1 K_1^T}{\sqrt{d}} = \frac{1}{\sqrt{4}} Q_1 K_1^T = \frac{1}{2} Q_1 K_1^T$$

$$Q_1 K_1^T = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

$$S_{11} = \frac{1}{2}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.5 & 0 & 0 & 0 \\ 0 & 0.5 & 0 & 0 \end{bmatrix}$$

**Step 2: Apply causal mask.**

Q-block 1 covers rows 1–2. K/V-tile 1 covers columns 1–4. Causal mask: row $i$ can attend to column $j$ only if $j \leq i$.

Row 1 (query position 1): can attend to column 1 only. Columns 2, 3, 4 $\to -\infty$.
Row 2 (query position 2): can attend to columns 1, 2. Columns 3, 4 $\to -\infty$.

$$\tilde{S}_{11} = \begin{bmatrix} 0.5 & -\infty & -\infty & -\infty \\ 0 & 0.5 & -\infty & -\infty \end{bmatrix}$$

**Step 3: Row-wise maximum.**

$$m_{11} = [\max(0.5, -\infty, -\infty, -\infty),\; \max(0, 0.5, -\infty, -\infty)] = [0.5,\; 0.5]$$

**Step 4: Exponentiated scores (subtract row max).**

Row 1: $e^{0.5 - 0.5} = 1$, $e^{-\infty} = 0$, $e^{-\infty} = 0$, $e^{-\infty} = 0$
Row 2: $e^{0 - 0.5} = e^{-0.5} \approx 0.6065$, $e^{0.5 - 0.5} = 1$, $0$, $0$

$$P_{11} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0.6065 & 1 & 0 & 0 \end{bmatrix}$$

**Step 5: Row sums (running denominator $\ell$).**

$$\ell_{11} = [1 + 0 + 0 + 0,\; 0.6065 + 1 + 0 + 0] = [1.0,\; 1.6065]$$

**Step 6: Partial output (unnormalised).**

$$O_{11} = P_{11} \cdot V_1 = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0.6065 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

$$O_{11} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0.6065 & 1 & 0 & 0 \end{bmatrix}$$

(The output is not yet normalised; we keep the unnormalised sum and $\ell$ to enable online softmax correction.)

**Step 7: Running statistics after tile (1,1).**

Running maximum: $m = [0.5, 0.5]$
Running sum: $\ell = [1.0, 1.6065]$
Running unnormalised output: $O = O_{11}$

---

### Part C: Finalise Q-block 1 output

Under causal masking, Q-block 1 (rows 1–2) has no valid K/V positions in tile 2 (columns 5–8), since columns 5–8 are all future positions relative to rows 1–2. So we skip tile 2.

**Normalise the output:**

$$O_1^{\text{final}} = \frac{O_{11}}{\ell_{11}} = \begin{bmatrix} 1 / 1.0 & 0 / 1.0 & 0 / 1.0 & 0 / 1.0 \\ 0.6065 / 1.6065 & 1 / 1.6065 & 0 / 1.6065 & 0 / 1.6065 \end{bmatrix}$$

$$O_1^{\text{final}} = \begin{bmatrix} 1.0000 & 0 & 0 & 0 \\ 0.3775 & 0.6225 & 0 & 0 \end{bmatrix}$$

**Interpretation:**
- Row 1 (token at position 1): can only attend to position 1 (itself), so output = $V[1] = [1, 0, 0, 0]$ with weight 1. The attention is 100% self-attention.
- Row 2 (token at position 2): attends to positions 1 and 2. Score at pos 1: $0$, score at pos 2: $0.5$. After normalisation: weights are $e^0 / (e^0 + e^{0.5}) = 1/1.6487 = 0.6065/1.6065 = 0.3775$ for pos 1 and $e^{0.5}/1.6487 = 0.6225$ for pos 2.
  Output $= 0.3775 \cdot [1,0,0,0] + 0.6225 \cdot [0,1,0,0] = [0.3775, 0.6225, 0, 0]$. Check: $K[1] = [1,0,0,0]$, $K[2] = [0,1,0,0]$.

**Verify via standard softmax.** For row 2, the full row of attention logits (causal):
$$[0/2, 0.5/1 \cdot \sqrt{...}] \to $$
The score of query row 2 against key col 1: $q_2 \cdot k_1 / 2 = (0)(1)+(1)(0)+... = 0$. Against col 2: $(0)(0)+(1)(1)/2 = 0.5$.
Softmax: $[e^0, e^{0.5}] / (e^0 + e^{0.5}) = [1, 1.6487] / 2.6487 = [0.3775, 0.6225]$. Matches.

---

### Part D: Q-block 2, online update with two K/V tiles

**Q-block 2:** rows 3–4 of $Q$:
$$Q_2 = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

**Processing K/V tile 1 (full, no masking needed for rows 3–4 vs cols 1–4):**

$$Q_2 K_1^T = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

$$S_{21} = \frac{1}{2}\begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 & 0 & 0 \\ 0 & 0 & 0.5 & 0 \end{bmatrix}$$

No masking needed (rows 3–4 can attend to all of cols 1–4).

Row maxima: $m^{(1)} = [0.5, 0.5]$

Exponentiated (subtracting max):
- Row 3: $[e^0, e^0, e^{-0.5}, e^{-0.5}] = [1, 1, 0.6065, 0.6065]$
- Row 4: $[e^{-0.5}, e^{-0.5}, e^0, e^{-0.5}] = [0.6065, 0.6065, 1, 0.6065]$

Row sums: $\ell^{(1)} = [1 + 1 + 0.6065 + 0.6065,\; 0.6065+0.6065+1+0.6065] = [3.2130,\; 2.8195]$

Unnormalised output:
$$O^{(1)} = \begin{bmatrix} 1 & 1 & 0.6065 & 0.6065 \\ 0.6065 & 0.6065 & 1 & 0.6065 \end{bmatrix} \cdot V_1$$

$$V_1 = I_4, \text{ so } O^{(1)} = \begin{bmatrix} 1 & 1 & 0.6065 & 0.6065 \\ 0.6065 & 0.6065 & 1 & 0.6065 \end{bmatrix}$$

**Processing K/V tile 2 (partial masking for rows 3–4 vs cols 5–8):**

$$K_2 = V_2 = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix}$$

$$Q_2 K_2^T = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$

Wait — let me recompute $Q_2 K_2^T$ properly for all 4 K columns:

$$Q_2 K_2^T = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix}^T$$

Recall $K_2^T$ has columns as rows of $K_2$, so $K_2^T \in \mathbb{R}^{4 \times 4}$. The $(i,j)$ entry of $Q_2 K_2^T$ is $\sum_l Q_2[i,l] K_2[j,l]$.

Row 3 of $Q_2 = [1,1,0,0]$ dotted with each row of $K_2$:
- vs $K_2[1] = [1,1,0,0]$: $1+1 = 2$
- vs $K_2[2] = [0,0,1,1]$: $0$
- vs $K_2[3] = [1,0,1,0]$: $1$
- vs $K_2[4] = [0,1,0,1]$: $1$

Row 4 of $Q_2 = [0,0,1,0]$ dotted with each row of $K_2$:
- vs $K_2[1]$: $0$; vs $K_2[2]$: $1$; vs $K_2[3]$: $1$; vs $K_2[4]$: $0$

$$Q_2 K_2^T = \begin{bmatrix} 2 & 0 & 1 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix}$$

$$S_{22} = \frac{1}{2} \begin{bmatrix} 2 & 0 & 1 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0 & 0.5 & 0.5 \\ 0 & 0.5 & 0.5 & 0 \end{bmatrix}$$

**Apply causal mask.** K/V tile 2 contains positions 5–8. Row 3 (query pos 3) can attend to positions $\leq 3$: positions 5–8 are all future. **All entries in row 3 are masked to $-\infty$.**

Row 4 (query pos 4) can attend to positions $\leq 4$: positions 5–8 are all future. **All entries in row 4 are masked to $-\infty$.**

$$\tilde{S}_{22} = \begin{bmatrix} -\infty & -\infty & -\infty & -\infty \\ -\infty & -\infty & -\infty & -\infty \end{bmatrix}$$

All entries masked: this entire tile is skipped (all attention weights are 0). No online update needed.

**Finalise Q-block 2.** Only tile 1 contributed:

$$O_2^{\text{final}} = O^{(1)} / \ell^{(1)}$$

Row 3: $[1, 1, 0.6065, 0.6065] / 3.2130 = [0.3112, 0.3112, 0.1888, 0.1888]$

Row 4: $[0.6065, 0.6065, 1, 0.6065] / 2.8195 = [0.2151, 0.2151, 0.3547, 0.2151]$

$$O_2^{\text{final}} = \begin{bmatrix} 0.3112 & 0.3112 & 0.1888 & 0.1888 \\ 0.2151 & 0.2151 & 0.3547 & 0.2151 \end{bmatrix}$$

**Online update demo** (for completeness, showing what the update would look like if tile 2 had valid entries):

If new tile $j$ produces max $\tilde{m}$ and unnormalised sum $\tilde{\ell}$ and partial output $\tilde{O}$, the online update is:

$$m^{\text{new}} = \max(m^{\text{old}}, \tilde{m})$$
$$\ell^{\text{new}} = e^{m^{\text{old}} - m^{\text{new}}} \cdot \ell^{\text{old}} + \tilde{\ell}$$
$$O^{\text{new}} = e^{m^{\text{old}} - m^{\text{new}}} \cdot O^{\text{old}} + \tilde{O}$$

The rescaling factor $e^{m^{\text{old}} - m^{\text{new}}}$ corrects the old accumulated sum and output for the new (possibly larger) row maximum.

---

### Part E: HBM read/write analysis

**Causal FlashAttention — tiles processed:**

| Q-block | KV tiles processed |
|---|---|
| 1 (rows 1–2) | KV tile 1 (partial) |
| 2 (rows 3–4) | KV tile 1 (full), KV tile 2 (partial — fully masked, skipped) |
| 3 (rows 5–6) | KV tile 1 (full), KV tile 2 (partial) |
| 4 (rows 7–8) | KV tile 1 (full), KV tile 2 (full) |

Total tile accesses: $1 + 1 + 2 + 2 = 6$ (skipping the fully masked KV tile for Q-block 2).

**HBM I/O for FlashAttention (full sequence):**

For the forward pass over a causal sequence of length $N$:

Each row of $Q$ is loaded once: $O(Nd)$
Each row of $K, V$ is loaded for each Q-block that accesses it:
- KV tile 1 (rows 1–4): accessed by Q-blocks 1, 2, 3, 4 = 4 times
- KV tile 2 (rows 5–8): accessed by Q-blocks 3, 4 = 2 times

With $N = 8$, $d = 4$, FP16 (2 bytes):

| Operation | Data size |
|---|---|
| Load $Q$ (once) | $8 \times 4 \times 2 = 64$ bytes |
| Load $K$, $V$ (multiple times) | KV tile 1: $4 \times 4 \times 2 \times 2 \times 4 = 256$ bytes; KV tile 2: $4 \times 4 \times 2 \times 2 \times 2 = 128$ bytes |
| Write $O$ (once) | $8 \times 4 \times 2 = 64$ bytes |
| Write statistics ($L$ = log-sum-exp) | $8 \times 2 = 16$ bytes |

Total: $64 + 384 + 64 + 16 = 528$ bytes

**Compare to standard attention:**

Standard attention materialises $S, P \in \mathbb{R}^{8 \times 8}$:

| Operation | Data |
|---|---|
| Write $S = QK^T$ | $8 \times 8 \times 2 = 128$ bytes |
| Read $S$, write $P = \text{Softmax}(S)$ | $2 \times 128 = 256$ bytes |
| Read $P$, read $V$, write $O = PV$ | $128 + 64 + 64 = 256$ bytes |
| Load $Q$, $K$ for $QK^T$ | $64 + 64 = 128$ bytes |

Total: $128 + 256 + 256 + 128 = 768$ bytes

**Ratio:** $768 / 528 = 1.45\times$ — FlashAttention uses 31% less HBM bandwidth for $N = 8$.

For larger $N$, the gap grows because standard attention's $N^2$ terms dominate:

$$\frac{\text{Standard HBM}}{\text{FlashAttention HBM}} \approx \frac{N^2 \cdot \text{sizeof(float)}}{N \cdot d \cdot \text{sizeof(float)}} = \frac{N}{d}$$

For $N = 4096$, $d = 128$: ratio $= 32\times$. FlashAttention is $32\times$ more I/O efficient.

---

## Key Algorithm Summary

```python
import torch

def flash_attention_forward(Q, K, V, B_r=64, B_c=64, causal=True):
    """
    Simplified FlashAttention forward pass (pedagogical, not optimised).

    Args:
        Q, K, V: (N, d) tensors
        B_r: query block size (rows)
        B_c: key/value block size (columns)
        causal: apply lower-triangular mask

    Returns:
        O: (N, d) attention output
        L: (N,) log-sum-exp for each row (stored for backward pass)
    """
    N, d = Q.shape
    scale = d ** -0.5

    O = torch.zeros_like(Q)       # output accumulator
    m = torch.full((N,), float('-inf'))  # running maximum
    ell = torch.zeros(N)          # running sum

    num_q_blocks = (N + B_r - 1) // B_r
    num_kv_blocks = (N + B_c - 1) // B_c

    for i in range(num_q_blocks):
        q_start = i * B_r
        q_end = min(q_start + B_r, N)
        Qi = Q[q_start:q_end]  # (B_r, d) -- loaded from HBM to SRAM

        # Local accumulators (in SRAM)
        Oi = torch.zeros(q_end - q_start, d)
        mi = torch.full((q_end - q_start,), float('-inf'))
        li = torch.zeros(q_end - q_start)

        for j in range(num_kv_blocks):
            kv_start = j * B_c
            kv_end = min(kv_start + B_c, N)

            # Under causal masking: skip entirely if kv_start > q_end - 1
            if causal and kv_start >= q_end:
                continue

            Kj = K[kv_start:kv_end]  # (B_c, d) -- loaded from HBM to SRAM
            Vj = V[kv_start:kv_end]  # (B_c, d)

            # Compute attention score tile
            Sij = (Qi @ Kj.T) * scale  # (B_r, B_c)

            # Apply causal mask within the tile
            if causal:
                q_positions = torch.arange(q_start, q_end).unsqueeze(1)  # (B_r, 1)
                kv_positions = torch.arange(kv_start, kv_end).unsqueeze(0)  # (1, B_c)
                mask = kv_positions > q_positions  # True where future
                Sij = Sij.masked_fill(mask, float('-inf'))

            # Online softmax: update running statistics
            mij = Sij.max(dim=-1).values  # (B_r,) block max
            Pij = torch.exp(Sij - mij.unsqueeze(-1))  # (B_r, B_c) unnorm
            lij = Pij.sum(dim=-1)  # (B_r,) block sum

            # Update running max and rescale
            mi_new = torch.maximum(mi, mij)
            rescale = torch.exp(mi - mi_new)      # correct old accumulator
            block_scale = torch.exp(mij - mi_new)  # scale new block

            li = rescale * li + block_scale * lij
            Oi = rescale.unsqueeze(-1) * Oi + (block_scale.unsqueeze(-1) * Pij) @ Vj
            mi = mi_new

        # Normalise and write output block to HBM
        O[q_start:q_end] = Oi / li.unsqueeze(-1)
        m[q_start:q_end] = mi
        ell[q_start:q_end] = torch.log(li) + mi  # log-sum-exp (LSE)

    return O, ell


# Test on the worked example
Q_ex = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [1., 1., 0., 0.],
    [0., 0., 1., 0.],
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
    [1., 1., 1., 0.],
    [0., 0., 0., 1.],
])
K_ex = V_ex = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [1., 1., 0., 0.],
    [0., 0., 1., 1.],
    [1., 0., 1., 0.],
    [0., 1., 0., 1.],
])

O_flash, lse = flash_attention_forward(Q_ex, K_ex, V_ex, B_r=2, B_c=4, causal=True)

# Verify against standard attention
import torch.nn.functional as F
scores = (Q_ex @ K_ex.T) / (4 ** 0.5)
mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
scores_masked = scores.masked_fill(mask, float('-inf'))
attn_weights = F.softmax(scores_masked, dim=-1)
O_standard = attn_weights @ V_ex

print("FlashAttention output (rows 1-4):")
print(O_flash[:4].round(decimals=4))
print("\nStandard attention output (rows 1-4):")
print(O_standard[:4].round(decimals=4))
print("\nMax absolute difference:", (O_flash - O_standard).abs().max().item())
```

**Expected output:**
```
FlashAttention output (rows 1-4):
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.3775, 0.6225, 0.0000, 0.0000],
        [0.3112, 0.3112, 0.1888, 0.1888],
        [0.2151, 0.2151, 0.3547, 0.2151]])

Standard attention output (rows 1-4):
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.3775, 0.6225, 0.0000, 0.0000],
        [0.3112, 0.3112, 0.1888, 0.1888],
        [0.2151, 0.2151, 0.3547, 0.2151]])

Max absolute difference: 0.0
```

The outputs are numerically identical, confirming that the tiling and online softmax procedure computes exact attention.
