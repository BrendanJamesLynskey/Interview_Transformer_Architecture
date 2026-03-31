# Implementing Transformer Attention from Scratch: Guide and Interview Questions

This guide covers the design decisions, common mistakes, and testing strategies for implementing Transformer attention components from scratch in PyTorch. Interviewers at top companies frequently ask candidates to implement these components live, so understanding the "why" behind each decision is as important as getting the code right.

---

## Conceptual Foundation

### Q1. Before writing code, what is the high-level purpose of attention, and what problem does it solve?

**Answer.**

A language model processes a sequence of tokens. To generate the next token, the model needs to aggregate information from previous positions in the sequence — but not all positions are equally relevant. A simple weighted sum over all positions (a bag-of-words) loses position-specific information. A fixed-size hidden state (like an LSTM) forgets early tokens.

Attention solves this by computing a **content-dependent weighted average** over the sequence. For each query position, it computes a relevance score against all key positions, normalises those scores into weights, and computes a weighted sum of the values at all positions.

The key innovation: the weights are computed dynamically from the content, so the model can selectively attend to whichever positions are most relevant for the current computation.

Mathematically:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q \in \mathbb{R}^{T_q \times d_k}$, $K \in \mathbb{R}^{T_k \times d_k}$, $V \in \mathbb{R}^{T_k \times d_v}$.

---

### Q2. Why is the dot product scaled by $\sqrt{d_k}$?

**Answer.**

For a query vector $q$ and key vector $k$ with components drawn i.i.d. from $\mathcal{N}(0, 1)$, the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has:

$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

So the standard deviation of the dot product is $\sqrt{d_k}$. Without scaling, as $d_k$ grows (say $d_k = 64$ or $d_k = 128$), the dot products grow large in magnitude. Large values pushed through softmax produce very sharp distributions: one attention weight near 1 and all others near 0. This corresponds to nearly one-hot attention, which:

1. Produces near-zero gradients for all "losing" keys/values, slowing learning.
2. Makes the output a nearly hard lookup rather than a soft aggregation.
3. Is numerically unstable in fp16/bf16.

Dividing by $\sqrt{d_k}$ restores the standard deviation to ~1, giving softmax inputs with a reasonable range and gradient flow.

---

### Q3. What is the difference between self-attention and cross-attention?

**Answer.**

In **self-attention**, queries, keys, and values all come from the same sequence:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

Every position attends to all other positions in the same sequence. Self-attention is used in both encoder and decoder blocks.

In **cross-attention**, the queries come from one sequence (the decoder) and the keys and values come from another sequence (the encoder output):

$$Q = X_\text{dec} W_Q, \quad K = X_\text{enc} W_K, \quad V = X_\text{enc} W_V$$

The decoder positions query into the encoder representations to incorporate source information. Cross-attention is the mechanism that "links" the encoder and decoder in encoder-decoder architectures (e.g., original Transformer, T5, Whisper).

Decoder-only models (GPT, Llama) use only self-attention — there is no separate encoder.

---

## Design Decisions

### Q4. What is multi-head attention and why is it better than single-head attention?

**Answer.**

Multi-head attention (MHA) runs $h$ attention heads in parallel, each with its own learned $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ projections, then concatenates the outputs and applies a final projection $W_O$:

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

$$\text{head}_i = \text{Attn}(X W_Q^{(i)},\; X W_K^{(i)},\; X W_V^{(i)})$$

Each head projects into a lower-dimensional space: if $d_\text{model} = 512$ and $h = 8$, each head dimension is $d_k = d_v = 64$.

**Why multiple heads are better:**

1. **Different attention patterns.** Different heads can attend to different types of relationships simultaneously — one head might track syntactic dependencies, another might track coreference, another might track positional proximity. A single head must average these patterns, losing resolution.

2. **Representation diversity.** The $h$ different projections allow the model to attend to $h$ different "subspaces" of the input simultaneously. This is analogous to having multiple convolutional filters in a CNN, each detecting different features.

3. **Empirical performance.** Single-head attention consistently underperforms multi-head at the same parameter count in practice, suggesting the inductive bias of multiple heads is beneficial.

---

### Q5. Walk through the shape transformations in multi-head attention.

**Answer.**

Starting with input $X \in \mathbb{R}^{B \times T \times d_\text{model}}$ (batch, sequence length, model dimension):

```
Input:  X             [B, T, d_model]

Project:
  Q = X @ W_Q         [B, T, d_model]  where W_Q: [d_model, d_model]
  K = X @ W_K         [B, T, d_model]
  V = X @ W_V         [B, T, d_model]

Reshape into heads:
  Q -> [B, T, h, d_k] -> transpose -> [B, h, T, d_k]
  K -> [B, T, h, d_k] -> transpose -> [B, h, T, d_k]
  V -> [B, T, h, d_v] -> transpose -> [B, h, T, d_v]

  where d_k = d_v = d_model / h

Attention scores:
  scores = Q @ K.transpose(-2,-1)     [B, h, T, T]
  scores = scores / sqrt(d_k)         [B, h, T, T]
  scores = softmax(scores, dim=-1)    [B, h, T, T]  (after masking)

Weighted sum of values:
  out = scores @ V                    [B, h, T, d_v]

Recombine heads:
  out -> transpose(1,2) -> [B, T, h, d_v]
  out -> reshape -> [B, T, d_model]  (h * d_v = d_model)

Output projection:
  out = out @ W_O                     [B, T, d_model]
```

The critical operation is the reshape that splits $d_\text{model}$ into $h$ heads of size $d_k$. This can be done with `.view()` or `.reshape()` in PyTorch — but `.contiguous()` may be needed after `.transpose()` before `.view()`.

---

### Q6. What are causal masking and padding masking, and how are they applied?

**Answer.**

**Causal masking (autoregressive masking):**

Prevents position $i$ from attending to positions $j > i$. This is essential for decoder-side language modelling: the model must not "see the future" during training. Without this mask, the model would trivially learn to copy the next token rather than predict it.

Implementation: create an upper triangular mask (including the diagonal) of $-\infty$ values, added to the attention scores before softmax:

```python
# For a sequence of length T:
mask = torch.triu(torch.ones(T, T), diagonal=1)  # Upper triangle, excluding diagonal
mask = mask.masked_fill(mask == 1, float('-inf'))
scores = scores + mask  # scores: [B, h, T, T]
```

After softmax, positions with $-\infty$ logits receive 0 attention weight.

**Padding masking:**

Language models process batches of sequences padded to the same length with a special `[PAD]` token. The model should not attend to padding positions (they contain no real information and are implementation artefacts).

Implementation: given a boolean padding mask `pad_mask` of shape $[B, T]$ where `True` indicates a padding position:

```python
# Expand to [B, 1, 1, T] to broadcast across heads and query positions
pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
scores = scores.masked_fill(pad_mask, float('-inf'))
```

**Important detail:** when masking softmax inputs, always use $-\infty$ (or a very large negative number like $-10^9$ for fp16). Using 0 does not produce zero attention weight — softmax(0) = 1/T, not 0.

---

## Common Mistakes

### Q7. What are the most common bugs when implementing attention from scratch?

**Answer.**

**Bug 1: Wrong dimension for scaling.**

```python
# Wrong: scales by d_model instead of d_k
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_model)

# Correct: scale by d_k = d_model / h
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
```

**Bug 2: Forgetting to apply causal mask before softmax.**

If the mask is applied after softmax, the attention distribution is already formed — masking it out produces incorrect renormalisation.

```python
# Wrong order:
attn_weights = F.softmax(scores, dim=-1)
attn_weights = attn_weights.masked_fill(mask, 0)  # Probabilities don't sum to 1!

# Correct order:
scores = scores.masked_fill(mask, float('-inf'))
attn_weights = F.softmax(scores, dim=-1)
```

**Bug 3: Using `.view()` on a non-contiguous tensor.**

After `.transpose()`, the tensor is non-contiguous in memory. Calling `.view()` on a non-contiguous tensor raises a RuntimeError.

```python
# Wrong: may raise RuntimeError
x = x.transpose(1, 2).view(B, T, d_model)

# Correct: make contiguous first
x = x.transpose(1, 2).contiguous().view(B, T, d_model)
# Or use reshape (handles non-contiguous tensors automatically)
x = x.transpose(1, 2).reshape(B, T, d_model)
```

**Bug 4: Wrong softmax dimension.**

Softmax must be applied over the key dimension (the last dimension of the $[B, h, T_q, T_k]$ score matrix), not the query dimension.

```python
# Wrong: normalises over queries instead of keys
attn_weights = F.softmax(scores, dim=-2)

# Correct: normalises over keys
attn_weights = F.softmax(scores, dim=-1)
```

**Bug 5: Numerical instability with fp16.**

Softmax with large inputs overflows in fp16. Cast scores to fp32 before softmax, then cast back.

```python
# Safer approach for mixed precision:
attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
```

**Bug 6: Output projection on wrong shape.**

The output projection $W_O$ expects the concatenated head dimension, not the per-head dimension.

```python
# Wrong: projecting a single head's output
out = self.out_proj(out)  # If out is [B, h, T, d_v], this is wrong shape

# Correct: concatenate heads first, then project
out = out.transpose(1, 2).reshape(B, T, self.d_model)  # [B, T, h*d_v = d_model]
out = self.out_proj(out)  # [B, T, d_model]
```

---

## Testing Strategies

### Q8. How do you verify that your attention implementation is correct?

**Answer.**

**Test 1: Compare against PyTorch's reference implementation.**

```python
import torch
import torch.nn.functional as F

# Your implementation
out_custom = my_attention(Q, K, V)

# PyTorch reference
out_ref = F.scaled_dot_product_attention(Q, K, V)

assert torch.allclose(out_custom, out_ref, atol=1e-5), \
    f"Max diff: {(out_custom - out_ref).abs().max()}"
```

**Test 2: Verify causal masking.**

A causally masked model should assign zero attention to future positions. After training or at initialisation, the attention weights at position $i$ should have exactly zero weight for all positions $j > i$.

```python
# After computing attention weights (before the value multiplication)
for i in range(T):
    for j in range(i+1, T):
        assert attn_weights[..., i, j].abs().max() < 1e-6, \
            f"Non-zero attention from pos {i} to future pos {j}"
```

**Test 3: Verify padding mask.**

Pad the last $k$ tokens of a sequence and check that attention weights to those positions are zero.

```python
# Create sequence with last 3 positions padded
pad_mask = torch.zeros(1, T, dtype=torch.bool)
pad_mask[0, -3:] = True

out_padded = my_attention(Q, K, V, key_padding_mask=pad_mask)
# Attending to padded positions should contribute nothing
assert attn_weights[..., -3:].abs().max() < 1e-6
```

**Test 4: Gradient flow.**

Verify that gradients flow through the attention weights to both Q and V projections.

```python
Q.requires_grad_(True)
V.requires_grad_(True)
out = my_attention(Q, K, V)
out.sum().backward()
assert Q.grad is not None and Q.grad.abs().max() > 0
assert V.grad is not None and V.grad.abs().max() > 0
```

**Test 5: Equivariance check for self-attention.**

Self-attention is permutation equivariant (without positional encoding): permuting the input should permute the output in the same way.

```python
perm = torch.randperm(T)
out_orig = my_attention(X, X, X)
out_perm = my_attention(X[:, perm], X[:, perm], X[:, perm])
assert torch.allclose(out_orig[:, perm], out_perm, atol=1e-5)
```

---

## Numerical Precision

### Q9. What precision issues arise in attention, and how are they handled in production?

**Answer.**

**Issue 1: Softmax overflow in fp16.**

fp16 has a maximum representable value of 65,504. For $d_k = 64$, the unscaled dot product of two unit vectors is $\sqrt{d_k} = 8$, which is safe. But in practice, after training, attention logits can exceed 100 in magnitude, causing softmax inputs to overflow when added to the $-10^4$ mask values.

Solution: flash attention and modern kernels cast to fp32 for the softmax computation and accumulate in fp32.

**Issue 2: Attention sink accumulation.**

Transformer models trained autoregressive often develop "attention sinks" — positions (typically the first token) that accumulate large attention weights as a form of garbage collection. This creates very sharp attention distributions that approach one-hot behaviour, amplifying numerical issues.

**Issue 3: The online softmax trick.**

Computing the standard softmax requires two passes over the sequence: one to find the max (for numerical stability) and one to compute the normalised exponentials. FlashAttention computes attention in tiles by maintaining running statistics (max and sum) that allow single-pass softmax computation in SRAM.

Standard numerically stable softmax:
$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

Online / tiled softmax (used in FlashAttention):
- Maintain running maximum $m^{(k)}$ and sum $s^{(k)}$ as you process blocks.
- Update: $m^{(k+1)} = \max(m^{(k)}, \max(x_\text{new block}))$
- Rescale accumulated sum when maximum changes.

**Production recommendation:** use `torch.nn.functional.scaled_dot_product_attention()` (PyTorch 2.0+), which automatically dispatches to FlashAttention when available and handles all precision issues correctly.
