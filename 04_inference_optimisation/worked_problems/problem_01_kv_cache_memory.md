# Worked Problem 01 — KV Cache Memory Calculation

## Problem Statement

Calculate the KV cache memory requirements for two models across a range of
sequence lengths and batch sizes.

**Models:**

| Model | Layers $L$ | KV heads $H_{kv}$ | Head dimension $d_h$ | Architecture |
|-------|-----------|-------------------|----------------------|--------------|
| LLaMA-2 7B | 32 | 32 | 128 | Multi-Head Attention (MHA) |
| LLaMA-2 70B | 80 | 8 | 128 | Grouped-Query Attention (GQA) |

**Precision:** BF16 (2 bytes per element) for both models.

**Configurations to evaluate:**

| Sequence length $T$ | Batch size $B$ |
|--------------------|----------------|
| 2,048 | 1, 8, 32 |
| 4,096 | 1, 8, 32 |
| 8,192 | 1, 8, 32 |

---

## Background: KV Cache Formula

At any point during decoding, the KV cache stores:
- The **key** tensor: for each layer, each KV head, and each position up to $T$.
- The **value** tensor: same shape.

The memory for one sequence's full KV cache at sequence length $T$ is:

$$
M_\text{KV}(B=1) = 2 \times L \times H_{kv} \times d_h \times T \times \text{bytes}
$$

The factor 2 accounts for keys and values. For a batch of $B$ sequences:

$$
M_\text{KV}(B) = B \times 2 \times L \times H_{kv} \times d_h \times T \times \text{bytes}
$$

**Units.** We express results in gigabytes (GB), where 1 GB $= 10^9$ bytes. Note
that GPU memory is typically reported in gibibytes (GiB $= 2^{30}$ bytes) by
vendors, so results in GB will appear slightly smaller than the display on tools
like `nvidia-smi`. We use GB throughout for clean arithmetic.

---

## Part 1 — LLaMA-2 7B (MHA)

**Parameters:**
- $L = 32$
- $H_{kv} = 32$ (MHA: KV heads equals query heads)
- $d_h = 128$
- bytes $= 2$ (BF16)

**Memory per token per sequence (constant across $T$):**

$$
m_\text{per token} = 2 \times L \times H_{kv} \times d_h \times 2 \text{ bytes}
= 2 \times 32 \times 32 \times 128 \times 2
$$

$$
= 2 \times 32 \times 32 \times 128 \times 2
= 2 \times 262{,}144
= 524{,}288 \text{ bytes}
= 0.524 \text{ MB per token per sequence}
$$

**Memory formula:** $M = B \times T \times 524{,}288$ bytes $= B \times T \times 0.524\,\text{MB}$

### Results Table — LLaMA-2 7B

| $T$ | $B$ | $B \times T$ tokens | Memory (GB) | Fits on 80GB GPU? |
|-----|-----|---------------------|-------------|-------------------|
| 2,048 | 1 | 2,048 | $2048 \times 0.524\,\text{MB} = 1.07\,\text{GB}$ | Yes |
| 2,048 | 8 | 16,384 | $16384 \times 0.524\,\text{MB} = 8.59\,\text{GB}$ | Yes |
| 2,048 | 32 | 65,536 | $65536 \times 0.524\,\text{MB} = 34.4\,\text{GB}$ | Yes (model weights ~14 GB leaves ~66 GB) |
| 4,096 | 1 | 4,096 | $4096 \times 0.524\,\text{MB} = 2.15\,\text{GB}$ | Yes |
| 4,096 | 8 | 32,768 | $32768 \times 0.524\,\text{MB} = 17.2\,\text{GB}$ | Yes |
| 4,096 | 32 | 131,072 | $131072 \times 0.524\,\text{MB} = 68.7\,\text{GB}$ | Marginal (model weights + KV ~83 GB) |
| 8,192 | 1 | 8,192 | $8192 \times 0.524\,\text{MB} = 4.29\,\text{GB}$ | Yes |
| 8,192 | 8 | 65,536 | $65536 \times 0.524\,\text{MB} = 34.4\,\text{GB}$ | Yes |
| 8,192 | 32 | 262,144 | $262144 \times 0.524\,\text{MB} = 137.4\,\text{GB}$ | No (exceeds 80 GB by $\sim 70$ GB) |

**Detailed calculation for $T = 4096$, $B = 32$:**

$$
M = 32 \times 32 \times 32 \times 128 \times 4096 \times 2
= 32 \times 2^{25} \times 2
= 32 \times 67{,}108{,}864
= 2{,}147{,}483{,}648 \text{ bytes}
\approx 2.15 \text{ GB} \times 32 = 68.7 \text{ GB}
$$

---

## Part 2 — LLaMA-2 70B (GQA)

**What is GQA?** Grouped-Query Attention (Ainslie et al., 2023) reduces the
number of KV heads relative to query heads. LLaMA-2 70B uses 64 query heads
but only 8 KV heads — each KV head is shared by 8 query heads. This reduces
KV cache memory by $64 / 8 = 8\times$ relative to MHA.

**Parameters:**
- $L = 80$
- $H_{kv} = 8$ (GQA: 64 query heads grouped into 8 KV groups)
- $d_h = 128$
- bytes $= 2$ (BF16)

**Memory per token per sequence:**

$$
m_\text{per token} = 2 \times 80 \times 8 \times 128 \times 2
= 2 \times 163{,}840
= 327{,}680 \text{ bytes}
= 0.328 \text{ MB per token per sequence}
$$

Despite having $80/32 = 2.5\times$ more layers, the 70B model has
$8/32 = 0.25\times$ the KV heads, yielding a net KV memory per token of:

$$
\frac{0.328}{0.524} \approx 0.626\times \text{ the 7B cost}
$$

The 70B model is cheaper per token than the 7B model — this is a deliberate
engineering choice: GQA allows the 70B to serve longer sequences or larger
batches per GPU (though the model weights themselves are much larger at $\sim 140$ GB).

### Results Table — LLaMA-2 70B

| $T$ | $B$ | $B \times T$ tokens | Memory (GB) |
|-----|-----|---------------------|-------------|
| 2,048 | 1 | 2,048 | $2048 \times 0.328\,\text{MB} = 0.671\,\text{GB}$ |
| 2,048 | 8 | 16,384 | $16384 \times 0.328\,\text{MB} = 5.37\,\text{GB}$ |
| 2,048 | 32 | 65,536 | $65536 \times 0.328\,\text{MB} = 21.5\,\text{GB}$ |
| 4,096 | 1 | 4,096 | $4096 \times 0.328\,\text{MB} = 1.34\,\text{GB}$ |
| 4,096 | 8 | 32,768 | $32768 \times 0.328\,\text{MB} = 10.7\,\text{GB}$ |
| 4,096 | 32 | 131,072 | $131072 \times 0.328\,\text{MB} = 43.0\,\text{GB}$ |
| 8,192 | 1 | 8,192 | $8192 \times 0.328\,\text{MB} = 2.69\,\text{GB}$ |
| 8,192 | 8 | 65,536 | $65536 \times 0.328\,\text{MB} = 21.5\,\text{GB}$ |
| 8,192 | 32 | 262,144 | $262144 \times 0.328\,\text{MB} = 85.9\,\text{GB}$ |

**Detailed calculation for $T = 8192$, $B = 8$:**

$$
M = 8 \times 2 \times 80 \times 8 \times 128 \times 8192 \times 2
$$

Breaking this down:
$$
= 8 \text{ (batch)} \times 2 \text{ (K+V)} \times 80 \text{ (layers)} \times 8 \text{ (KV heads)} \times 128 \text{ (d\_h)} \times 8192 \text{ (tokens)} \times 2 \text{ (bytes)}
$$
$$
= 8 \times 2 \times 80 \times 8 \times 128 \times 8192 \times 2
= 8 \times 2{,}684{,}354{,}560
= 21{,}474{,}836{,}480 \text{ bytes}
\approx 21.5 \text{ GB} \checkmark
$$

---

## Part 3 — Side-by-Side Comparison

**Memory at $B=8$, varying sequence length:**

| Sequence length | 7B KV cache | 70B KV cache | 70B / 7B ratio |
|----------------|-------------|--------------|----------------|
| 2,048 | 8.59 GB | 5.37 GB | 0.63 |
| 4,096 | 17.2 GB | 10.7 GB | 0.62 |
| 8,192 | 34.4 GB | 21.5 GB | 0.63 |

The ratio is constant (as expected from the per-token analysis) at approximately
$0.63$. GQA's KV reduction more than compensates for the 70B model's greater depth.

---

## Part 4 — Practical Implications

### Maximum batch size on 80 GB GPU (decode only, single GPU slice)

For the 7B model, model weights in BF16 occupy $\approx 14$ GB, leaving
$\sim 66$ GB for the KV cache:

$$
B_\text{max}(T) = \left\lfloor \frac{66 \times 10^9}{T \times 524{,}288} \right\rfloor
$$

| $T$ | $B_\text{max}$ (7B on 80 GB) |
|-----|-------------------------------|
| 512 | 246 sequences |
| 2,048 | 61 sequences |
| 4,096 | 30 sequences |
| 8,192 | 15 sequences |

For the 70B model, weights occupy $\approx 140$ GB — it does not fit on a single
80 GB GPU regardless of KV cache. A 4-GPU (or 2-GPU with tensor parallelism)
deployment is required.

### Quantisation impact on KV cache

Quantising the KV cache to INT8 (1 byte per element) halves all values in the
tables above. INT4 (0.5 bytes per element) quarters them, enabling $4\times$
the batch size or sequence length. KV cache quantisation (e.g., FP8 KV in
vLLM, KIVI) is an active area of LLM serving research and is now supported in
production frameworks.

---

## Summary Formula

$$
\boxed{M_\text{KV} = B \times 2 \times L \times H_{kv} \times d_h \times T \times \text{bytes\_per\_element}}
$$

| Model | Per-token, per-sequence (BF16) |
|-------|-------------------------------|
| LLaMA-2 7B (MHA, 32 KV heads) | 0.524 MB |
| LLaMA-2 70B (GQA, 8 KV heads) | 0.328 MB |
