# Grouped Query Attention

## Overview

Grouped Query Attention (GQA) is an architectural modification that reduces the memory and bandwidth cost of the KV cache during inference with minimal quality loss. It is now standard in production-scale LLMs including LLaMA-3, Mistral, and Gemma.

---

## Fundamentals

### Q1. What is Multi-Head Attention (MHA) and why is the KV cache a memory bottleneck?

**Answer.**

In standard MHA with $H$ heads, each head $h$ independently computes:

$$Q_h = X W_h^Q, \quad K_h = X W_h^K, \quad V_h = X W_h^V$$
$$\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h$$
$$\text{MHA}(X) = \text{concat}(\text{head}_1, \ldots, \text{head}_H)\, W^O$$

where $X \in \mathbb{R}^{T \times d}$, $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d \times d_k}$, and $d_k = d/H$.

During autoregressive decoding, all previous K and V vectors must be available for the current step. These are stored in the **KV cache**: for each layer, $K \in \mathbb{R}^{T \times H \times d_k}$ and $V \in \mathbb{R}^{T \times H \times d_k}$.

**Memory formula.** For $L$ layers, $H$ heads, head dimension $d_k$, batch size $B$, and sequence length $T$:

$$\text{KV cache} = 2 \times L \times B \times T \times H \times d_k \times \text{sizeof(dtype)}$$

**Concrete example — LLaMA-2 70B** ($L=80$, $H=64$, $d_k=128$, FP16 = 2 bytes), single sequence, $T=4096$:
$$2 \times 80 \times 1 \times 4096 \times 64 \times 128 \times 2 \approx 10.7\text{ GB}$$

At batch size $B = 16$: $\approx 171$ GB — more than the model weights. The KV cache is the dominant inference memory cost for long sequences.

---

### Q2. What is Multi-Query Attention (MQA)?

**Answer.**

MQA (Shazeer, 2019) uses a single shared K head and V head across all query heads:

$$Q_h = X W_h^Q \quad (\text{H separate projections})$$
$$K = X W^K, \quad V = X W^V \quad (\text{1 shared projection each})$$
$$\text{head}_h = \text{softmax}\!\left(\frac{Q_h K^T}{\sqrt{d_k}}\right) V$$

**KV cache reduction:** $H \to 1$ KV head — a factor of $H$ reduction.

**Quality trade-off:** All heads share the same K and V, limiting diversity. Each head can ask a different question (via $Q_h$) but they all look at the same representation of the context. This limits representational power, and MQA shows meaningful quality degradation on tasks requiring diverse attention patterns, particularly at larger model scales.

---

### Q3. How does Grouped Query Attention work, and how does it generalise MHA and MQA?

**Answer.**

GQA (Ainslie et al., 2023) uses $G$ KV head groups, where $1 \leq G \leq H$. Query heads are partitioned into $G$ groups of size $H/G$. All query heads within a group share one K head and one V head.

$$Q_h = X W_h^Q \quad (H \text{ projections})$$
$$K_g = X W_g^K, \quad V_g = X W_g^V \quad (G \text{ projections})$$
$$\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_{g(h)}^T}{\sqrt{d_k}}\right) V_{g(h)}$$

where $g(h) = \lceil hG / H \rceil$ assigns query head $h$ to its group.

**Unification:**
- $G = H$: each head has its own K/V $\Rightarrow$ **MHA**
- $G = 1$: all heads share one K/V $\Rightarrow$ **MQA**
- $1 < G < H$: **GQA** proper

---

### Q4. Calculate the KV cache memory savings from GQA.

**Answer.**

**Per token, per layer:**
$$\text{MHA}: \quad 2 \times H \times d_k \times \text{sizeof(dtype)}$$
$$\text{GQA}: \quad 2 \times G \times d_k \times \text{sizeof(dtype)}$$

**Reduction factor:** $H / G$

**Worked example — Mistral 7B** ($H=32$ Q heads, $G=8$ KV heads, $d_k=128$, FP16):

| Quantity | MHA (hypothetical) | GQA (actual) |
|---|---|---|
| KV per token per layer | $2 \times 32 \times 128 \times 2 = 16$ KB | $2 \times 8 \times 128 \times 2 = 4$ KB |
| KV for $T=4096$, $L=32$, $B=1$ | $32 \times 4096 \times 16\text{ KB} = 2.0$ GB | $32 \times 4096 \times 4\text{ KB} = 0.5$ GB |
| KV at batch $B=16$ | $\sim 32$ GB | $\sim 8$ GB |

The $4\times$ reduction directly enables $4\times$ larger batch sizes or $4\times$ longer sequences at the same GPU memory budget.

---

## Intermediate

### Q5. How does GQA affect the parameter count of the model?

**Answer.**

The K and V projection matrices change from $H$ matrices of shape $d \times d_k$ to $G$ matrices. Q projections and the output projection $W^O$ are unchanged.

**Parameter change (attention weights only):**
$$\Delta\text{params} = 2 \times (H - G) \times d \times d_k$$

**Mistral 7B** ($d=4096$, $H=32$, $G=8$, $d_k=128$, $L=32$ layers):
$$\Delta = 2 \times 24 \times 4096 \times 128 \times 32 \approx 805M \text{ params}$$

This is roughly 11% of the 7B total. In practice, models that switch to GQA typically compensate by increasing the FFN hidden dimension slightly to match total parameter count, so the effective model capacity is preserved.

---

### Q6. Why does GQA retain most of MHA's quality while MQA degrades noticeably?

**Answer.**

The key insight is that attention head diversity is primarily driven by the **query projections**, not the key/value projections.

Each head $h$ computes $\text{softmax}(Q_h K_{g(h)}^T / \sqrt{d_k})$, producing a head-specific attention distribution over all positions. Even when heads within a group share $K_g$, the different $Q_h$ vectors produce completely different attention patterns — the softmax output is unique per head.

**Where quality is lost.** The shared $K_g$ must encode information useful for all $H/G$ query heads in the group. With $G = 1$ (MQA), one K projection must serve all $H$ heads — there is severe representational pressure in a $d_k$-dimensional space. With $G = H/8$ (e.g., GQA 4:1), each K projection serves only 4 query heads, with far less pressure.

**Empirical scaling.** Ainslie et al. (2023) found that GQA with $G = H/8$ matches MHA quality on most benchmarks when trained from scratch. The quality gap grows as $G$ decreases toward 1. Fine-tuning an existing MHA model to GQA recovers most quality with roughly $5\%$ additional pretraining compute.

---

### Q7. How do you convert a pretrained MHA model to GQA?

**Answer.**

Ainslie et al. (2023) show that mean-pooling the MHA K/V heads within each group gives a good initialisation:

$$W_g^{K,\text{GQA}} = \frac{G}{H} \sum_{h \in \text{group}(g)} W_h^{K,\text{MHA}}$$

The procedure:
1. Define the $G$ groups (e.g., for $H=32$, $G=8$: groups of 4 consecutive heads)
2. Mean-pool the K projection matrices within each group
3. Mean-pool the V projection matrices within each group
4. Keep Q projections and output projections unchanged
5. Fine-tune for $\sim 5\%$ of pretraining tokens on the same data distribution

**Why mean-pooling.** During MHA training, heads within the same positional group often learn similar coarse-grained patterns (due to training dynamics and the loss landscape). Mean-pooling produces a K/V that represents the average "view" of the group, which is a better starting point for fine-tuning than random initialisation or taking a single head's weights.

**Results.** After uptrain fine-tuning:
- GQA ($G = H/8$) recovers $>95\%$ of MHA quality on standard benchmarks
- MQA ($G = 1$) recovers $\sim 85\text{-}90\%$ — meaningful quality loss remains

---

## Advanced

### Q8. Analyse the throughput impact of GQA, distinguishing memory bandwidth from compute.

**Answer.**

Modern GPU inference at small batch sizes is **memory-bandwidth-limited**, not compute-limited. Each decode step loads:
- All model weight matrices from HBM
- The full KV cache from HBM

**Bandwidth per decode step (single token, one layer):**

$$\text{Model weights}: \quad (4d^2 + 2d \cdot d_{\text{ff}}) \times \text{sizeof(dtype)} \quad \text{(constant)}$$
$$\text{KV cache (MHA)}: \quad 2 \times H \times T \times d_k \times \text{sizeof(dtype)}$$
$$\text{KV cache (GQA)}: \quad 2 \times G \times T \times d_k \times \text{sizeof(dtype)}$$

For Mistral 7B ($d=4096$, $d_{\text{ff}}=14336$, $H=32$, $G=8$, $d_k=128$, FP16) at $T=1024$:

| Component | Memory (bytes) |
|---|---|
| Model weights per layer | $(4 \times 4096^2 + 2 \times 4096 \times 14336) \times 2 \approx 371$ MB |
| KV cache per layer (MHA, $T=1024$) | $2 \times 32 \times 1024 \times 128 \times 2 = 16$ MB |
| KV cache per layer (GQA, $T=1024$) | $2 \times 8 \times 1024 \times 128 \times 2 = 4$ MB |

At $T=1024$, the KV cache is $\sim 4\%$ of total memory — GQA gives modest speedup. At $T=32768$, the KV cache would be $\sim 512$ MB vs $\sim 128$ MB — then GQA becomes a significant bandwidth reduction.

**Practical throughput numbers (A100 80GB, Mistral 7B, batch 32):**
- GQA enables fitting longer sequences or larger batches into GPU memory
- At long sequences ($T > 8192$), GQA yields $1.5\text{--}2\times$ throughput improvement vs hypothetical MHA
- At short sequences ($T < 1024$), weight bandwidth dominates and GQA impact is $< 10\%$

---

### Q9. How does GQA interact with tensor parallelism across multiple GPUs?

**Answer.**

In standard tensor parallelism, attention heads are split across $P$ GPUs: each GPU handles $H/P$ Q heads and $H/P$ K/V heads.

With GQA ($G$ KV heads, $H$ Q heads), each GPU handles $H/P$ Q heads but only $G/P$ K/V heads. This requires $G \geq P$ and $G$ divisible by $P$.

**Failure case.** If $G = 4$ and $P = 8$: some GPUs would hold zero KV heads. The KV cache must be broadcast to all GPUs, adding communication that partially cancels the memory savings.

**Design implication.** GQA group count $G$ must be chosen to be divisible by all target parallelism levels:

| Target parallelism | Minimum $G$ |
|---|---|
| 1, 2 | $G \geq 2$, divisible by $2$ |
| 1, 2, 4 | $G \geq 4$, divisible by $4$ |
| 1, 2, 4, 8 | $G = 8$ (or multiple of 8) |

This is precisely why Mistral 7B uses $G = 8$: it supports 1, 2, 4, and 8-way tensor parallelism without extra communication — aligning with standard A100/H100 8-GPU server configurations.

**General rule.** Set $G = \text{lcm}(P_1, P_2, \ldots)$ where $P_i$ are the intended GPU counts, or more conservatively $G = 8$ as the standard.
