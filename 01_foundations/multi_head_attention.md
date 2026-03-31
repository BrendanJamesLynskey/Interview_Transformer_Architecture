# Multi-Head Attention

## Overview

Multi-head attention (MHA) is the mechanism that allows Transformers to simultaneously attend to information from different representation subspaces. Understanding why multiple heads are used, how the projection matrices relate, how parameters are counted, and how $d_{\text{model}}$, $d_k$, $d_v$, and $h$ interact is essential for both interviews and practical implementation work.

---

## Tier 1: Fundamentals

### Q1. What is multi-head attention and why is it used instead of single-head attention?

**Answer.**

**Single-head attention** computes one set of attention weights over the entire $d_{\text{model}}$-dimensional representation, producing a single weighted combination of values. This means the model can only learn one kind of relationship between tokens per layer.

**Multi-head attention** runs $h$ attention operations in parallel, each in a lower-dimensional subspace:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

where:

$$\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

**Why multiple heads?**

Each head operates on a different learned linear projection of the input. Different heads can specialise in capturing different types of relationships simultaneously within the same layer:

- Head 1 might focus on local syntactic patterns (attending to adjacent tokens)
- Head 2 might capture coreference (pronouns attending to noun referents)
- Head 3 might track positional offsets (every token attending to position $i-2$)
- Head 4 might aggregate global information

A single head with dimension $d_{\text{model}}$ would attempt to blend all of these into one attention map, which is more restrictive. Multiple heads in lower-dimensional subspaces provide a form of ensemble diversity within a layer.

**Important constraint:** The total computation is designed to be equal. With $h$ heads and $d_k = d_v = d_{\text{model}} / h$, the FLOPs for MHA match those of single-head attention with dimension $d_{\text{model}}$.

---

### Q2. Describe the four projection matrices $W^Q$, $W^K$, $W^V$, $W^O$. What are their shapes and roles?

**Answer.**

For head $i$ in a model with $d_{\text{model}}$, $h$ heads, $d_k = d_v = d_{\text{model}} / h$:

**Per-head projections (for head $i$):**

| Matrix | Shape | Role |
|---|---|---|
| $W_i^Q$ | $d_{\text{model}} \times d_k$ | Projects input to query subspace for head $i$ |
| $W_i^K$ | $d_{\text{model}} \times d_k$ | Projects input to key subspace for head $i$ |
| $W_i^V$ | $d_{\text{model}} \times d_v$ | Projects input to value subspace for head $i$ |

These projections allow each head to "look at" a different aspect of the input. The linear projection is what gives MHA its expressiveness — without it, all heads would compute identical attention (since they'd use the same representation space).

**Output projection:**

| Matrix | Shape | Role |
|---|---|---|
| $W^O$ | $h \cdot d_v \times d_{\text{model}}$ | Projects concatenated head outputs back to $d_{\text{model}}$ |

After concatenation, the combined vector has dimension $h \cdot d_v = d_{\text{model}}$ (when $d_v = d_{\text{model}} / h$). The $W^O$ projection mixes information across heads and maps back to the residual stream dimension.

**Implementation note:** In practice, all $h$ query projections are batched into a single matrix $W^Q \in \mathbb{R}^{d_{\text{model}} \times h d_k} = \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ and then reshaped to run all heads in parallel. This is identical mathematically but more efficient on hardware.

---

### Q3. What is the relationship between $d_{\text{model}}$, $d_k$, $d_v$, and the number of heads $h$?

**Answer.**

The standard design choice from "Attention Is All You Need" (Vaswani et al., 2017) is:

$$d_k = d_v = \frac{d_{\text{model}}}{h}$$

**Rationale for this choice:**

1. **Computational equivalence:** The total FLOPs for attention with $h$ heads of dimension $d_k = d_{\text{model}}/h$ equals single-head attention with dimension $d_{\text{model}}$. The $h$-fold reduction in per-head dimension exactly compensates for running $h$ heads.

2. **Parameter budget consistency:** Each head's $W_i^Q$ is $d_{\text{model}} \times d_k = d_{\text{model}} \times (d_{\text{model}}/h)$. Across all $h$ heads, total parameters in $W^Q$ is $h \times d_{\text{model}} \times (d_{\text{model}}/h) = d_{\text{model}}^2$ — same as a single $d_{\text{model}} \times d_{\text{model}}$ projection.

3. **Residual stream compatibility:** With $d_v = d_{\text{model}} / h$, concatenating $h$ heads gives $h \cdot d_v = d_{\text{model}}$, matching the residual stream dimension before the $W^O$ projection.

**Constraints:**
- $h$ must divide $d_{\text{model}}$ evenly
- Typical choices: (h=8, $d_{\text{model}}$=512), (h=12, $d_{\text{model}}$=768), (h=16, $d_{\text{model}}$=1024), (h=96, $d_{\text{model}}$=12288)

**Non-standard variants:**
- Multi-Query Attention (MQA): $d_k = d_v = d_{\text{model}}/h$ but only 1 set of K, V projections shared across all heads
- Grouped-Query Attention (GQA): $g$ groups of query heads share K, V projections; $g < h$

---

### Q4. Walk through the complete multi-head attention computation step by step, specifying tensor shapes at each stage.

**Answer.**

Given: $d_{\text{model}} = 512$, $h = 8$, $d_k = d_v = 64$, sequence length $n = 10$, batch size $B = 1$.

**Input:** $X \in \mathbb{R}^{B \times n \times d_{\text{model}}} = \mathbb{R}^{1 \times 10 \times 512}$

**Step 1: Linear projections**

Batched projection matrices (all heads combined):
- $W^Q \in \mathbb{R}^{512 \times 512}$, $W^K \in \mathbb{R}^{512 \times 512}$, $W^V \in \mathbb{R}^{512 \times 512}$

$$Q = X W^Q \in \mathbb{R}^{1 \times 10 \times 512}$$
$$K = X W^K \in \mathbb{R}^{1 \times 10 \times 512}$$
$$V = X W^V \in \mathbb{R}^{1 \times 10 \times 512}$$

**Step 2: Reshape into heads**

Split the last dimension into $h$ heads of size $d_k$:

$$Q \to \mathbb{R}^{1 \times 10 \times 8 \times 64} \to \mathbb{R}^{8 \times 10 \times 64}$$

(Transpose batch and head dims, merge batch with head for parallel computation)

**Step 3: Scaled dot-product attention per head**

For each head $i$: $\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$

Scores: $Q_i K_i^T / \sqrt{64} \in \mathbb{R}^{10 \times 10}$

Output per head: $\in \mathbb{R}^{10 \times 64}$

All heads simultaneously: $\mathbb{R}^{8 \times 10 \times 64}$

**Step 4: Concatenate heads**

Reshape $\mathbb{R}^{8 \times 10 \times 64} \to \mathbb{R}^{1 \times 10 \times 512}$ (merge head dim with $d_v$)

**Step 5: Output projection**

$$\text{output} = \text{Concat}(heads) \cdot W^O \in \mathbb{R}^{1 \times 10 \times 512}$$

$W^O \in \mathbb{R}^{512 \times 512}$

**Final output shape:** $\mathbb{R}^{1 \times 10 \times 512}$ — same as input. The MHA block is shape-preserving.

---

## Tier 2: Intermediate

### Q5. Derive the total parameter count for one MHA layer. Compare with a single-head attention layer of equal dimension.

**Answer.**

**Multi-head attention parameters:**

For $d_{\text{model}} = 512$, $h = 8$, $d_k = d_v = 64$:

| Component | Shape | Parameters |
|---|---|---|
| $W^Q$ (all heads) | $512 \times 512$ | 262,144 |
| $W^K$ (all heads) | $512 \times 512$ | 262,144 |
| $W^V$ (all heads) | $512 \times 512$ | 262,144 |
| $W^O$ | $512 \times 512$ | 262,144 |
| **Total (no bias)** | | **1,048,576** |

In general: $4 d_{\text{model}}^2$ parameters for one MHA layer (ignoring biases).

**Single-head attention with $d_k = d_v = d_{\text{model}}$:**

| Component | Shape | Parameters |
|---|---|---|
| $W^Q$ | $512 \times 512$ | 262,144 |
| $W^K$ | $512 \times 512$ | 262,144 |
| $W^V$ | $512 \times 512$ | 262,144 |
| $W^O$ | $512 \times 512$ | 262,144 |
| **Total** | | **1,048,576** |

**The parameter counts are identical.** MHA achieves multi-head diversity at the same parameter cost by splitting the projection dimensionality: $h$ heads of $d_{\text{model}}/h$ dimensions each sum to the same total as one head of $d_{\text{model}}$ dimensions.

This is a key design property: you can change the number of heads (within the constraint that $h$ divides $d_{\text{model}}$) without changing the parameter count or computational cost.

---

### Q6. Implement multi-head attention from scratch in PyTorch. Verify it produces the same result as PyTorch's built-in implementation.

**Answer.**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation matching PyTorch's nn.MultiheadAttention
    (with batch_first=True, no bias for clarity).

    Args:
        d_model: Total model dimension
        num_heads: Number of attention heads; must divide d_model evenly
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # dimension per head

        # Combined projection matrices (all heads in one matmul)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, seq, d_model) -> (batch, heads, seq, d_k).
        """
        batch, seq, _ = x.shape
        # Reshape: (batch, seq, heads, d_k)
        x = x.view(batch, seq, self.num_heads, self.d_k)
        # Transpose to: (batch, heads, seq, d_k)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, heads, seq, d_k) -> (batch, seq, d_model).
        """
        batch, _, seq, _ = x.shape
        # Transpose to: (batch, seq, heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # Merge: (batch, seq, d_model)
        return x.view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,          # (batch, n_q, d_model)
        key: torch.Tensor,            # (batch, n_k, d_model)
        value: torch.Tensor,          # (batch, n_k, d_model)
        attn_mask: torch.Tensor = None,  # (n_q, n_k) or (batch*heads, n_q, n_k)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (batch, n_q, d_model)
            attn_weights: (batch, heads, n_q, n_k)
        """
        # 1. Project to Q, K, V
        Q = self.split_heads(self.W_q(query))   # (batch, heads, n_q, d_k)
        K = self.split_heads(self.W_k(key))     # (batch, heads, n_k, d_k)
        V = self.split_heads(self.W_v(value))   # (batch, heads, n_k, d_v)

        # 2. Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (batch, heads, n_q, n_k)

        if attn_mask is not None:
            # attn_mask: True = masked (will become -inf)
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                scores = scores + attn_mask   # Additive mask (already -inf values)

        attn_weights = F.softmax(scores, dim=-1)   # (batch, heads, n_q, n_k)

        context = torch.matmul(attn_weights, V)    # (batch, heads, n_q, d_k)

        # 3. Merge heads and project output
        context = self.merge_heads(context)        # (batch, n_q, d_model)
        output = self.W_o(context)                 # (batch, n_q, d_model)

        return output, attn_weights


# ── Verification against PyTorch built-in ────────────────────────────────────

torch.manual_seed(42)
d_model, num_heads, seq_len, batch = 64, 8, 10, 2

# Instantiate both
custom_mha = MultiHeadAttention(d_model, num_heads)
torch_mha = nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)

# Copy weights from custom to torch (torch uses a single in_proj_weight matrix)
# W_q, W_k, W_v are stacked in torch's in_proj_weight
with torch.no_grad():
    torch_mha.in_proj_weight.copy_(
        torch.cat([custom_mha.W_q.weight, custom_mha.W_k.weight, custom_mha.W_v.weight], dim=0)
    )
    torch_mha.out_proj.weight.copy_(custom_mha.W_o.weight)

# Forward pass
x = torch.randn(batch, seq_len, d_model)

custom_out, custom_weights = custom_mha(x, x, x)

# PyTorch MHA returns average attention weights by default
torch_out, _ = torch_mha(x, x, x, need_weights=False)

print(f"Custom output shape:  {custom_out.shape}")     # (2, 10, 64)
print(f"PyTorch output shape: {torch_out.shape}")      # (2, 10, 64)
print(f"Max abs difference:   {(custom_out - torch_out).abs().max().item():.2e}")
# Expected: < 1e-5 (small floating-point differences)

# Verify parameter count
total_params = sum(p.numel() for p in custom_mha.parameters())
print(f"Total parameters: {total_params:,}")  # Should be 4 * d_model^2 = 16,384
print(f"Expected (4 * {d_model}^2): {4 * d_model**2:,}")
```

**Expected output:**
```
Custom output shape:  torch.Size([2, 10, 64])
PyTorch output shape: torch.Size([2, 10, 64])
Max abs difference:   < 1e-05
Total parameters: 16,384
Expected (4 * 64^2): 16,384
```

---

### Q7. What is the difference between Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)? When would you use each?

**Answer.**

**Multi-Head Attention (MHA) — baseline:**
- $h$ query heads, $h$ key heads, $h$ value heads
- Full parameter count and KV cache

**Multi-Query Attention (MQA, Shazeer 2019):**
- $h$ query heads, but only 1 key head and 1 value head shared across all queries
- KV cache reduced by factor $h$
- Small quality degradation, large inference speedup (memory bandwidth bottleneck)
- Used in: PaLM, Falcon, early Starcoder

**Grouped-Query Attention (GQA, Ainslie et al. 2023):**
- $h$ query heads divided into $g$ groups; each group shares 1 K and 1 V head
- $g = 1$: reduces to MQA; $g = h$: reduces to MHA
- More flexible quality-efficiency tradeoff
- Used in: LLaMA-2 70B, Mistral 7B, Gemma, LLaMA-3

**Parameter comparison** ($d_{\text{model}} = 4096$, $h = 32$, $d_k = 128$):

| Variant | $W^K$ params | $W^V$ params | KV cache/token |
|---|---|---|---|
| MHA ($g=32$) | $4096 \times 4096$ = 16.7M | 16.7M | $2 \times 32 \times 128 = 8192$ floats |
| GQA ($g=8$) | $4096 \times 1024$ = 4.2M | 4.2M | $2 \times 8 \times 128 = 2048$ floats |
| MQA ($g=1$) | $4096 \times 128$ = 0.5M | 0.5M | $2 \times 1 \times 128 = 256$ floats |

**When to use:**
- **MHA:** When memory is not the bottleneck and you want maximum quality (e.g., encoder models, smaller models)
- **GQA:** Best practical trade-off for large decoder models with long contexts; $g = h/4$ is common
- **MQA:** Maximum inference throughput when KV cache is the hard constraint; quality loss is acceptable

---

## Tier 3: Advanced

### Q8. Explain the mechanistic interpretability perspective on attention heads. What do "induction heads" reveal about how Transformers learn in-context?

**Answer.**

**Induction heads (Olsson et al., 2022):**

One of the most concrete mechanistic findings about Transformer attention. An "induction circuit" consists of two cooperating heads across two layers:

**Layer 1: Previous token head.** Head attends to the token immediately preceding each position. This writes information about token $t-1$ into the residual stream at position $t$.

**Layer 2: Induction head.** At position $t$ with token $A$, this head searches for previous occurrences of $A$ in the context and attends to the token that followed the previous occurrence.

**Effect:** If the context contains `...[A][B]...[A][?]`, the induction head attends to the first $[B]$ and copies it to position $?$. This implements a form of "copy the completion of this pattern."

**Why this matters:**

1. **In-context learning:** Induction heads can implement approximate nearest-neighbour retrieval from the context. If the prompt contains several examples of a pattern, the induction head can generalise it to a new instance.

2. **Emergent behaviour:** The induction circuit appears spontaneously during training, not by design. It's a self-organised solution to pattern completion.

3. **Phase transitions:** Olsson et al. showed a sharp phase transition in training loss that coincides with the formation of induction heads — suggesting this is a critical capability threshold.

4. **Compositionality of heads:** The circuit requires cooperation between heads in different layers via the residual stream. This demonstrates that "what each head does" cannot be understood in isolation.

**Limitations of the mechanistic view:**

- Induction heads are well-characterised but most heads in large models remain poorly understood
- The superposition hypothesis (Elhage et al., 2022) suggests individual neurons and heads are polysemantic — encoding multiple features simultaneously — making clean circuits harder to find at scale
- Circuit analysis at scale (GPT-4 class models) is an active research frontier

---

### Q9. How would you modify the MHA computation to support Rotary Position Embeddings (RoPE)? Why does RoPE interact naturally with the attention dot product?

**Answer.**

**Motivation:** Standard sinusoidal positional encodings add position information to token embeddings before the entire Transformer stack. This means position information is entangled with token semantics from the first layer, and the interaction with Q, K projections is indirect.

RoPE (Su et al., 2021) instead encodes relative position directly into the dot product used to compute attention scores, by rotating Q and K vectors in 2D subspaces.

**RoPE construction:**

For a token at position $m$ with feature vector $x \in \mathbb{R}^{d_k}$, RoPE applies a rotation matrix $R_m$:

$$\tilde{q}_m = R_m q_m, \quad \tilde{k}_n = R_n k_n$$

where $R_m$ is a block-diagonal rotation matrix with blocks:

$$R_m^{(j)} = \begin{pmatrix} \cos(m \theta_j) & -\sin(m \theta_j) \\ \sin(m \theta_j) & \cos(m \theta_j) \end{pmatrix}, \quad \theta_j = 10000^{-2j/d_k}$$

**The key property:**

$$\tilde{q}_m^T \tilde{k}_n = (R_m q_m)^T (R_n k_n) = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n$$

The dot product depends only on the **relative position** $n - m$, not the absolute positions. This is the rotation property: $R_m^T R_n = R_{n-m}$.

**Implementation:**

```python
import torch
import math


def precompute_rope_frequencies(d_k: int, max_seq_len: int, base: int = 10000) -> tuple:
    """
    Precompute cos and sin values for RoPE.
    Returns: cos, sin each of shape (max_seq_len, d_k // 2)
    """
    # Frequencies: theta_j = base^{-2j/d_k}, j = 0, ..., d_k//2 - 1
    freqs = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k//2,)
    positions = torch.arange(max_seq_len).float()                       # (max_seq_len,)
    angles = torch.outer(positions, freqs)                              # (max_seq_len, d_k//2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to query or key tensor.

    Args:
        x:   (batch, heads, seq_len, d_k)
        cos: (seq_len, d_k // 2)
        sin: (seq_len, d_k // 2)
    Returns:
        (batch, heads, seq_len, d_k) with RoPE applied
    """
    seq_len, half_d = cos.shape
    # Split x into pairs of dimensions
    x1, x2 = x[..., :half_d], x[..., half_d:]  # each (batch, heads, seq_len, d_k//2)

    # Broadcast cos/sin over batch and head dims
    cos_ = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, d_k//2)
    sin_ = sin.unsqueeze(0).unsqueeze(0)

    # Rotation: [x1, x2] * cos + [-x2, x1] * sin
    rotated_x1 = x1 * cos_ - x2 * sin_
    rotated_x2 = x2 * cos_ + x1 * sin_

    return torch.cat([rotated_x1, rotated_x2], dim=-1)


# Usage in MHA forward pass:
# cos, sin = precompute_rope_frequencies(d_k, max_seq_len)
# cos_slice = cos[:seq_len]  # for current sequence length
# sin_slice = sin[:seq_len]
# Q_rotated = apply_rope(Q, cos_slice, sin_slice)
# K_rotated = apply_rope(K, cos_slice, sin_slice)
# scores = Q_rotated @ K_rotated.transpose(-2, -1) / sqrt(d_k)
```

**Advantages of RoPE:**

1. **Relative position naturalness:** The attention score inherently reflects relative position due to the rotation property
2. **Length generalisation:** RoPE extrapolates to longer sequences better than fixed absolute encodings (extended with YaRN, LongRoPE, etc.)
3. **No modification to value vectors:** Only Q and K are rotated; V remains unchanged
4. **Used by:** LLaMA (all versions), Mistral, Gemma, Falcon, most modern open models

**Comparison with ALiBi (Press et al., 2021):** ALiBi adds a linear position bias directly to the attention scores ($-m|i-j|$ for head $m$) rather than rotating Q/K. Simpler, but RoPE has become more dominant due to better performance on code and mathematical reasoning tasks.
