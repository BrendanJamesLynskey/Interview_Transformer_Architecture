# KV Cache Mechanism

## Overview

The KV cache is the primary memory optimisation in autoregressive LLM inference. Understanding what is cached, why, and how memory scales is essential for system design interviews and production ML engineering roles.

---

## Fundamentals

### Q1. Why is a KV cache needed in autoregressive decoding?

**Answer.**

In autoregressive decoding, the model generates one token at a time. To generate token $t$, it must compute attention over all tokens $1, 2, \ldots, t-1$ plus the new token $t$ itself.

Without caching, computing attention at step $t$ requires:
1. Recomputing queries $Q^{(t)}$, keys $K^{(t)}$, and values $V^{(t)}$ for all positions $1 \ldots t$
2. Computing the attention scores $Q^{(t)} (K^{(t)})^T$

This means re-processing all previous tokens at every step — $O(t)$ work per token, $O(T^2)$ total for a sequence of length $T$.

With the KV cache, at each step $t$:
- The keys $K_{1:t-1}$ and values $V_{1:t-1}$ from all previous tokens are stored in memory (the KV cache)
- Only the new token's query, key, and value are computed
- The new key $K_t$ and value $V_t$ are appended to the cache
- Attention is computed as $Q_t \cdot K_{1:t}^T$ (new query against all cached keys)

This reduces per-step work from $O(t \cdot d \cdot H)$ to $O(H \cdot d)$ for the projection, plus $O(t \cdot d \cdot H)$ for the attention computation (which is unavoidable — you must attend to all past tokens).

**What is not cached:** The query $Q_t$ is always freshly computed from the new token embedding, since it depends on the current token's content.

---

### Q2. Exactly what tensors are stored in the KV cache?

**Answer.**

For each transformer layer, the KV cache stores the key and value projections for all past sequence positions:

$$\text{KV cache layer } \ell = \{K^{(\ell)}_{1:t}, V^{(\ell)}_{1:t}\}$$

where $K^{(\ell)}_{1:t} \in \mathbb{R}^{t \times H \times d_k}$ and $V^{(\ell)}_{1:t} \in \mathbb{R}^{t \times H \times d_k}$.

**Dimensions:**
- $t$: current sequence length (grows by 1 each decode step)
- $H$: number of KV heads (equal to query heads in MHA, or fewer in GQA/MQA)
- $d_k$: head dimension ($= d / H_Q$ where $H_Q$ is the number of query heads)

**What is not stored:**
- Query projections $Q$: only needed momentarily to compute attention scores
- Attention weights (softmax outputs): recomputed each step from Q and cached K
- FFN activations: not needed for future steps
- Layer outputs: passed forward but not cached (they feed the next layer)

**Typical access pattern.** At decode step $t$:
1. New token embedding is computed
2. For each layer: compute $Q_t$, $K_t$, $V_t$ from the embedding
3. Append $K_t$ to layer's K cache; append $V_t$ to layer's V cache
4. Compute attention: $\text{softmax}(Q_t \cdot K_{1:t}^T / \sqrt{d_k}) \cdot V_{1:t}$
5. Continue to next layer

---

### Q3. How does KV cache memory scale with sequence length and batch size?

**Answer.**

For a model with $L$ layers, $H_{KV}$ KV heads, head dimension $d_k$, batch size $B$, and sequence length $T$:

$$\text{KV cache} = 2 \times L \times B \times T \times H_{KV} \times d_k \times \text{bytes per element}$$

The factor of 2 accounts for both K and V.

**Per-token cost** (independent of $B$, $T$):
$$\text{cost per token} = 2 \times L \times H_{KV} \times d_k \times \text{bytes}$$

This is the key quantity to memorise: KV cache grows **linearly** with sequence length and batch size. There is no quadratic dependence — the quadratic cost occurs in the attention computation (FLOPs), not the cache storage.

**Example — LLaMA-2 7B** ($L=32$, $H_{KV}=32$, $d_k=128$, FP16 = 2 bytes):
$$\text{per token} = 2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \text{ bytes} = 512 \text{ KB}$$

At $T = 4096$, $B = 1$: $4096 \times 512\text{ KB} = 2\text{ GB}$
At $T = 4096$, $B = 8$: $8 \times 2\text{ GB} = 16\text{ GB}$

---

## Intermediate

### Q4. What is the multi-layer KV cache and how is it organised in memory?

**Answer.**

Each of the $L$ transformer layers has its own independent KV cache. The layers do not share caches because the key and value representations differ at each layer (they are projections of different intermediate representations).

**Memory layout (contiguous per layer, per head):**

```
KV Cache Layout:
  Layer 0:
    K: [batch, seq, n_kv_heads, head_dim]  (e.g., shape [8, 4096, 32, 128])
    V: [batch, seq, n_kv_heads, head_dim]
  Layer 1: K, V (same shape)
  ...
  Layer L-1: K, V
```

**Access pattern implications.** At decode step $t$ in layer $\ell$:
- Write: append new K/V at position $t$ — one random write per layer (sequential with growing offset)
- Read: read all $K^{(\ell)}_{1:t}$ and $V^{(\ell)}_{1:t}$ — sequential read of $t$ rows

The reads are contiguous in memory if laid out as `[batch, seq, heads, dim]`, making memory bandwidth efficient. The total bytes read for one decode step across all layers:

$$2 \times L \times H_{KV} \times d_k \times t \times 2 \text{ bytes}$$

This is the KV cache bandwidth cost per decode step, which grows linearly with $t$ — the reason decode steps slow down over long sequences.

---

### Q5. Explain PagedAttention. What problem does it solve and how?

**Answer.**

**The problem: KV cache memory fragmentation.**

In naive implementations, each sequence's KV cache is pre-allocated as a contiguous block of maximum length. For a max context length of 4096 tokens:
- A sequence that uses only 100 tokens wastes memory for 3996 unused positions
- This internal fragmentation means GPU memory cannot be fully utilised
- You cannot easily share memory between sequences of different lengths

Additionally, sequences in a batch have unpredictable lengths (requests arrive at different times, complete at different times). Pre-allocating for maximum length is wasteful.

**PagedAttention** (Kwon et al., 2023; vLLM) borrows the virtual memory paging concept from OS design:

1. **Physical memory blocks.** KV cache memory is divided into fixed-size "pages" (blocks), each holding $B_{\text{page}}$ token positions (e.g., 16 tokens).

2. **Logical-to-physical mapping.** Each sequence has a logical KV sequence that is mapped to physical pages via a block table. Logical pages are contiguous; physical pages can be anywhere in memory.

3. **On-demand allocation.** Physical pages are allocated as needed. A sequence of 100 tokens uses approximately $\lceil 100/B_{\text{page}} \rceil$ pages, wasting at most $B_{\text{page}} - 1$ positions.

4. **Attention with indirection.** The attention kernel looks up the physical page address for each logical block and gathers key/value vectors. This adds a small indirection cost but eliminates fragmentation.

**Benefits:**
- Near-zero internal fragmentation (wasted $< 1$ page per sequence)
- Enables **copy-on-write** sharing for beam search (multiple beams share the same prompt KV cache pages until they diverge)
- Enables serving more sequences simultaneously on the same GPU
- Memory utilisation improves from $\sim 60\%$ (naive) to $\sim 90+\%$ (paged)

---

### Q6. How does prefill differ from decode in terms of KV cache behaviour?

**Answer.**

**Prefill phase.** The prompt tokens ($T_p$ tokens) are processed together in a single forward pass. This is essentially training-style execution:
- All $T_p$ queries, keys, and values are computed simultaneously
- Standard (FlashAttention) attention is computed over the full $T_p \times T_p$ matrix
- At the end of prefill, all $T_p$ K and V vectors are written to the KV cache
- KV cache population: $2 \times L \times T_p \times H_{KV} \times d_k$ bytes written in one pass

Prefill is compute-bound (matrix multiplications dominate). Time scales as $O(T_p^2)$ for the attention and $O(T_p)$ for other operations.

**Decode phase.** One token is generated at a time:
- At each step: compute Q, K, V for one token; append K, V to cache; compute attention against all cached tokens
- KV cache grows by $2 \times L \times H_{KV} \times d_k$ bytes per step

Decode is memory-bandwidth-bound (loading model weights and KV cache dominates). Time per step is roughly constant at $O(T_p + t_{\text{gen}})$ for the attention (growing with sequence length).

**The prefill-decode performance asymmetry.** Modern GPUs have high throughput for large matrix multiplications (prefill) but are bottlenecked by memory bandwidth for decode. Techniques like continuous batching and chunked prefill address the imbalance by mixing prefill and decode work to improve GPU utilisation.

---

## Advanced

### Q7. Calculate the KV cache memory for LLaMA-2 7B, 13B, and 70B at standard serving configurations.

**Answer.**

See also: `worked_problems/problem_01_kv_cache_memory.md` for detailed derivations.

**Model parameters:**

| Model | $L$ | $H_{KV}$ | $d_k$ | Notes |
|---|---|---|---|---|
| LLaMA-2 7B | 32 | 32 | 128 | MHA |
| LLaMA-2 13B | 40 | 40 | 128 | MHA |
| LLaMA-2 70B | 80 | 8 | 128 | GQA |

**Per-token KV cost** (FP16):
$$\text{7B}: 2 \times 32 \times 32 \times 128 \times 2 = 524{,}288 \approx 512\text{ KB/token}$$
$$\text{13B}: 2 \times 40 \times 40 \times 128 \times 2 = 819{,}200 \approx 800\text{ KB/token}$$
$$\text{70B}: 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \approx 320\text{ KB/token}$$

Note: 70B uses GQA ($H_{KV} = 8$ instead of $H_{KV} = 64$), making its per-token KV cost much lower than 13B despite having $6\times$ more layers.

**Total KV cache at different batch/sequence configurations (FP16):**

| Model | $B=1$, $T=2048$ | $B=8$, $T=2048$ | $B=32$, $T=4096$ |
|---|---|---|---|
| 7B | 1.0 GB | 8.0 GB | 64.0 GB |
| 13B | 1.6 GB | 12.8 GB | 102.4 GB |
| 70B | 0.6 GB | 4.9 GB | 39.4 GB |

The 70B model's KV cache at $B=32$, $T=4096$ is smaller than the 7B model's — a striking demonstration of GQA's impact.

---

### Q8. Why does the KV cache cause memory bandwidth to dominate decode performance? Quantify this.

**Answer.**

**GPU roofline model.** For an operation with arithmetic intensity $I$ (FLOPs/byte):
$$\text{time} = \max\left(\frac{\text{FLOPs}}{\text{compute peak}}, \frac{\text{bytes}}{{\text{memory BW}}}\right)$$

The crossover point (roofline) for A100 SXM: compute peak $= 312$ TFLOPS (BF16), memory BW $= 2$ TB/s.
Ridge point: $312 \times 10^{12} / (2 \times 10^{12}) = 156$ FLOPs/byte.

Any operation with $I < 156$ FLOPs/byte is memory-bandwidth-limited on A100.

**Arithmetic intensity of a single decode step (attention portion).**

At decode step $t$ with sequence $t$ tokens already cached:
- FLOPs: $2 \times H \times d_k \times t$ (one new Q, attention over $t$ K and V)
- Bytes read from HBM: $2 \times L \times H_{KV} \times d_k \times t \times 2$ bytes (read K and V cache)

For LLaMA-2 7B at $t = 1024$, $H = H_{KV} = 32$, $d_k = 128$, $L = 32$:
$$I_{\text{attention}} = \frac{2 \times 32 \times 128 \times 1024}{2 \times 32 \times 32 \times 128 \times 1024 \times 2} = \frac{1}{64} \approx 0.016 \text{ FLOPs/byte}$$

This is $10{,}000\times$ below the A100 ridge point. Attention during decode is extremely memory-bandwidth-limited.

**Linear layer arithmetic intensity** (also in the decode step):
$$I_{\text{linear}} = \frac{2 \times d^2}{d^2 \times 2} = 1 \text{ FLOPs/byte} \quad \text{(weight size equals FLOP count for batch 1)}$$

This is also memory-bandwidth-limited (1 vs 156 FLOPs/byte).

**Conclusion.** The entire decode step is bottlenecked by HBM bandwidth — both the KV cache reads and the weight matrix reads. Adding more compute (tensor cores) does not help; only faster memory bandwidth or techniques that reduce the data loaded per step (quantisation, GQA) improve decode throughput.
