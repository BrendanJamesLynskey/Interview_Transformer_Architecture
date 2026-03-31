# FlashAttention

## Overview

FlashAttention (Dao et al., 2022) is an I/O-aware algorithm for computing exact attention that dramatically reduces memory usage and improves wall-clock speed by exploiting the GPU memory hierarchy. It does not approximate attention — it produces identical results to standard attention while being faster and more memory-efficient.

---

## Fundamentals

### Q1. What is the standard attention memory complexity and why is it a problem?

**Answer.**

Standard attention computes:
$$S = QK^T \in \mathbb{R}^{N \times N}, \quad P = \text{Softmax}(S), \quad O = PV$$

where $Q, K, V \in \mathbb{R}^{N \times d}$ and $N$ is the sequence length.

**Memory complexity.** The attention matrix $S$ (and $P$) has $N^2$ entries. Each entry is a float (2 or 4 bytes):
$$\text{Memory}(S) = N^2 \times \text{sizeof(dtype)}$$

**Concrete examples:**

| Sequence length | FP16 attention matrix |
|---|---|
| $N = 1024$ | $2$ MB |
| $N = 4096$ | $32$ MB |
| $N = 16384$ | $512$ MB |
| $N = 65536$ | $8$ GB |
| $N = 131072$ | $32$ GB |

For $N = 65536$ (64k tokens), the attention matrix alone requires $8$ GB — exceeding one A100's HBM for a single head. With $H = 32$ heads and $L = 32$ layers during backpropagation (which requires storing all attention matrices), this is completely infeasible.

The problem is not just memory capacity — it is also I/O. Reading and writing $N^2$ values to and from HBM (High Bandwidth Memory, the GPU's main DRAM) takes time proportional to $N^2$, independent of compute speed.

---

### Q2. What is the GPU memory hierarchy and why does it matter for attention?

**Answer.**

Modern GPUs have a two-level memory hierarchy:

| Memory | Size (A100) | Bandwidth | Latency |
|---|---|---|---|
| SRAM (on-chip, shared memory / registers) | $\sim 20$ MB per GPU | $\sim 19$ TB/s | $\sim 1$ cycle |
| HBM (off-chip, "GPU memory") | $40$–$80$ GB | $\sim 2$ TB/s | $\sim 100$ cycles |

**The bottleneck.** Most GPU kernels spend more time waiting for HBM reads/writes than doing arithmetic. The ratio of compute to memory bandwidth (arithmetic intensity) for standard attention is low: you do $O(N^2 d)$ FLOPs but transfer $O(N^2)$ bytes for the attention matrix — giving arithmetic intensity $\sim d$ FLOPs/byte, which is low for typical $d = 64\text{–}128$.

**The opportunity.** If an algorithm can keep all intermediate results in SRAM (fast memory) and only read/write $O(N)$ data to HBM, the total HBM transfers drop from $O(N^2)$ to $O(N)$, giving a dramatic wall-clock speedup even without reducing FLOPs.

---

### Q3. At a high level, what is FlashAttention's algorithm strategy?

**Answer.**

FlashAttention's strategy is **tiling with online softmax**:

1. **Tile.** Divide $Q$, $K$, $V$ into blocks small enough to fit in SRAM (typically $B_r$ rows for $Q$ blocks and $B_c$ columns for $K$/$V$ blocks).
2. **Incremental softmax.** Process blocks sequentially. Maintain running statistics (running max and running sum) to compute the correct softmax without ever materialising the full $N \times N$ attention matrix.
3. **Accumulate output.** Incrementally accumulate the output $O$ block-by-block, correcting for the running softmax statistics.

**Result:** The full attention output is computed correctly without ever writing the $N \times N$ attention matrix to HBM. HBM reads/writes are $O(N)$ for $Q$, $K$, $V$, and $O$, plus $O(N)$ for the running statistics.

---

## Intermediate

### Q4. Explain the online softmax trick. Why is it needed for tiling?

**Answer.**

**The problem.** Standard softmax over a row of $A \in \mathbb{R}^N$ requires the full row:
$$\text{Softmax}(A)_j = \frac{e^{A_j}}{\sum_{k=1}^N e^{A_k}}$$

This requires reading all $N$ elements before computing any output. If $A$ is split into blocks, we cannot compute the final softmax for block 1 without seeing blocks 2 through $N/B_c$.

**Stable softmax trick.** Subtract the row maximum for numerical stability:
$$\text{Softmax}(A)_j = \frac{e^{A_j - m}}{\sum_{k=1}^N e^{A_k - m}} \quad \text{where } m = \max_k A_k$$

**Online update.** When processing block $i$ and then block $i+1$, we can update the running statistics:

After block $i$: running max $m_i = \max_{k \leq i \cdot B_c} A_k$, running denominator $\ell_i = \sum_{k=1}^{i \cdot B_c} e^{A_k - m_i}$

After seeing block $i+1$ with new max $\tilde{m} = \max_{k \in \text{block } i+1} A_k$:

$$m_{i+1} = \max(m_i, \tilde{m})$$
$$\ell_{i+1} = e^{m_i - m_{i+1}} \ell_i + \sum_{k \in \text{block }i+1} e^{A_k - m_{i+1}}$$

The factor $e^{m_i - m_{i+1}}$ rescales the old sum to the new max.

**Output accumulation.** Similarly, the output $O = PV$ can be accumulated incrementally. After seeing block $j$ of $K/V$:

$$O_{\text{new}} = \frac{\ell_{\text{old}}}{\ell_{\text{new}}} e^{m_{\text{old}} - m_{\text{new}}} O_{\text{old}} + \frac{\text{block softmax weights}}{\ell_{\text{new}}} V_{\text{block}}$$

The rescaling factor $e^{m_{\text{old}} - m_{\text{new}}}$ corrects for the updated maximum. This allows exact attention with $O(N)$ HBM I/O.

---

### Q5. Derive the memory and FLOPs complexity of FlashAttention vs standard attention.

**Answer.**

**Standard attention:**

| Operation | HBM I/O | FLOPs |
|---|---|---|
| $S = QK^T$ | Read $Q, K$: $O(Nd)$; Write $S$: $O(N^2)$ | $O(N^2 d)$ |
| $P = \text{Softmax}(S)$ | Read, write $S$: $O(N^2)$ | $O(N^2)$ |
| $O = PV$ | Read $P$: $O(N^2)$, read $V$: $O(Nd)$, write $O$: $O(Nd)$ | $O(N^2 d)$ |
| **Total** | $O(N^2 + Nd)$ | $O(N^2 d)$ |

For large $N$, HBM I/O is dominated by $N^2$.

**FlashAttention:**

All intermediate attention scores are computed and discarded within SRAM. Only final outputs and running statistics are written to HBM.

| What is written to HBM | Size |
|---|---|
| Output $O$ | $O(Nd)$ |
| Running max $m$ (for backprop) | $O(N)$ |
| Running log-sum-exp $\ell$ (for backprop) | $O(N)$ |

| Quantity | FlashAttention |
|---|---|
| HBM I/O | $O(Nd)$ (reads: $Q, K, V$; writes: $O$, stats) |
| FLOPs | $O(N^2 d)$ — identical to standard |
| Memory | $O(N)$ — only $O$ and statistics |

**Conclusion:** FlashAttention has the same FLOPs as standard attention but $O(N)$ HBM I/O instead of $O(N^2)$. Since modern GPUs are bandwidth-limited for attention computation, this translates directly to wall-clock speedup proportional to the I/O reduction factor $\sim N/d$.

---

### Q6. What are the wall-clock speedups and memory savings of FlashAttention in practice?

**Answer.**

**Memory savings.** For a single attention layer, FP16:

| $N$ | Standard attention matrix | FlashAttention overhead |
|---|---|---|
| 1024 | 2 MB | $\sim 8$ KB (stats only) |
| 8192 | 128 MB | $\sim 64$ KB |
| 65536 | 8 GB | $\sim 512$ KB |

The $O(N^2) \to O(N)$ reduction is transformative for long sequences. FlashAttention makes contexts of $32\text{k}\text{--}128\text{k}$ tokens feasible.

**Wall-clock speedup (FlashAttention paper, A100):**
- $N = 1024$: $\sim 2\times$ faster than PyTorch standard attention
- $N = 4096$: $\sim 4\times$ faster
- $N = 16384$: $\sim 7\times$ faster

Speedup increases with $N$ because the standard algorithm becomes increasingly bandwidth-bound.

**Enabling long context.** FlashAttention was the key enabler for practical long-context models:
- LLaMA-1/2: trained at $4096$ tokens (pre-FlashAttention or early FA)
- LLaMA-3: $8192\text{k}$ tokens (with FA-2)
- Many models: $128\text{k+}$ contexts using FA-2/FA-3

---

### Q7. What improvements does FlashAttention-2 introduce?

**Answer.**

FlashAttention-2 (Dao, 2023) improves on FA-1 in three main areas:

**1. Fewer non-matmul FLOPs.**
FA-1 spent a disproportionate number of operations on rescaling (the running max/sum corrections). FA-2 reorganises the algorithm to reduce these non-matmul operations (which run at much lower throughput than tensor core matrix multiplications on modern GPUs). Result: $\sim 2\times$ speedup on A100.

**2. Better parallelism across the sequence dimension.**
FA-1 parallelised across batch and head dimensions but was sequential across the sequence for a single head. FA-2 adds parallelism over the query sequence dimension, allowing better GPU occupancy for long sequences with a small number of heads (common in models with GQA).

**3. Better work partitioning between warps.**
FA-1 had suboptimal warp-level communication. FA-2 redesigns the tiling to minimise warp synchronisation, improving SM (streaming multiprocessor) utilisation.

**Quantitative improvement (A100 FP16, causal attention):**

| Sequence length | FA-1 (TFLOPS/s) | FA-2 (TFLOPS/s) | Fraction of peak A100 |
|---|---|---|---|
| 1024 | $\sim 100$ | $\sim 170$ | $\sim 55\%$ |
| 4096 | $\sim 130$ | $\sim 200$ | $\sim 65\%$ |
| 16384 | $\sim 140$ | $\sim 210$ | $\sim 68\%$ |

FA-2 reaches $\sim 70\%$ of theoretical peak A100 throughput — exceptionally high for a real kernel.

---

## Advanced

### Q8. How does FlashAttention handle the backward pass? What statistics must be stored?

**Answer.**

The backward pass for attention requires recomputing the attention matrix $P$ (to compute gradients). Standard backpropagation stores $P$ during the forward pass ($O(N^2)$ memory). FlashAttention uses **recomputation**: $P$ is not stored; instead, it is recomputed from $Q$, $K$, and the stored statistics.

**Stored statistics:** For each row $i$ of the attention matrix, FlashAttention stores:
$$L_i = m_i + \log(\ell_i)$$

the log-sum-exp (LSE) of row $i$. This is $O(N)$ storage.

**Backward recomputation.** During the backward pass:
1. Load a tile of $Q_i$ from HBM
2. Load a tile of $K_j$, $V_j$ from HBM
3. Recompute $S_{ij} = Q_i K_j^T / \sqrt{d}$
4. Recover $P_{ij} = \exp(S_{ij} - L_i)$ (correct attention probabilities without re-running softmax)
5. Compute local gradients $dQ_i$, $dK_j$, $dV_j$
6. Accumulate gradients and write back

**Total backward HBM I/O:** $O(Nd)$ reads of $Q, K, V, O, dO$; $O(N)$ reads of LSE statistics; $O(Nd)$ writes of $dQ, dK, dV$.

**Cost of recomputation.** The FLOPs for the backward pass are roughly $2\times$ the forward pass (because the attention matrix is computed twice). This is a worthwhile trade: the memory saving of $O(N^2)$ is far more valuable than the extra compute in bandwidth-limited regimes.

---

### Q9. What is the tiling strategy for FlashAttention? Derive the block size constraints.

**Answer.**

FlashAttention tiles $Q$ into blocks of $B_r$ rows and $K, V$ into blocks of $B_c$ rows. For each $Q$ tile, it iterates over all $K/V$ tiles, accumulating the output in SRAM.

**SRAM footprint for one iteration:**
- $Q$ tile: $B_r \times d$ elements
- $K$ tile: $B_c \times d$ elements
- $V$ tile: $B_c \times d$ elements
- Output accumulator: $B_r \times d$ elements
- Statistics ($m$, $\ell$): $2 \times B_r$ elements (negligible)

**Total:** $(2B_r + 2B_c) \times d$ elements (in FP16: $\times 2$ bytes)

**Constraint.** This must fit in SRAM ($M$ bytes):
$$(2B_r + 2B_c) \times d \times 2 \leq M$$

**Optimal block size.** To maximise arithmetic intensity (FLOPs per HBM byte), we want to maximise the tile sizes. Subject to the SRAM constraint:

$$B_r = B_c = B \Rightarrow 4Bd \times 2 \leq M \Rightarrow B \leq \frac{M}{8d}$$

**Concrete example.** A100 SRAM $M = 192$ KB $= 196608$ bytes, $d = 128$:

$$B \leq \frac{196608}{8 \times 128} = 192 \text{ rows per block}$$

So tiles of $192 \times 128$ elements fit in SRAM. The algorithm processes all $\lceil N/192 \rceil^2$ tile pairs without writing intermediate results to HBM.

**Causal masking in tiling.** For causal (autoregressive) attention, $Q$ tile $i$ only attends to $K$ tiles $j \leq i$. When $j < i$, the tile is fully valid (no masking needed). When $j = i$, the tile is on the diagonal and requires masking within the tile (set upper-triangular entries to $-\infty$). Tiles with $j > i$ are entirely skipped, halving the work for causal models (saving both compute and HBM I/O).

---

### Q10. What are the implications of FlashAttention for long-context LLM training and inference?

**Answer.**

**Training implications:**

1. **Sequence length ceiling raised.** Before FA, training at $N > 4096$ was impractical for most models (memory exhausted by attention matrices). FA enables $N = 32768\text{–}131072$ on A100 80GB.

2. **Batch size improvement.** Freed attention matrix memory can be reallocated to larger batches, improving hardware utilisation and gradient noise (smaller variance).

3. **No approximation.** Unlike sparse attention (Longformer, BigBird) or linear attention, FlashAttention computes exact softmax attention. Quality is identical to standard attention — there is no accuracy/speed trade-off, only speed/speed trade-off (wall-clock vs FLOPs).

4. **Gradient checkpoint integration.** FA's built-in recomputation aligns naturally with activation checkpointing, allowing further memory reduction at the cost of extra compute.

**Inference implications:**

1. **Prefill speedup.** During prompt processing (prefill), the full $N \times N$ attention matrix must be computed. FA reduces time quadratically for long prompts — a 64k-token prompt runs $\sim 10\text{--}15\times$ faster with FA-2 vs standard attention.

2. **Decode step.** During single-token decode, the attention matrix is $1 \times T$ (one query against all cached keys). This is small and already memory-bandwidth-bound in a different way (loading the KV cache dominates). FA provides smaller benefits here, but still improves SM occupancy.

3. **Long-context inference.** FA makes 128k-token inference tractable without out-of-memory errors, enabling retrieval-augmented generation, document analysis, and agentic workloads over long contexts.
