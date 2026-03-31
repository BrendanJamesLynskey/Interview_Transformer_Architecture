# Continuous Batching and Inference Scheduling

## Overview

Serving large language models efficiently requires fundamentally different
batching strategies from traditional deep learning inference. Because token
generation is autoregressive — each token depends on all previous tokens — the
output length of a request is unknown at arrival time and varies by orders of
magnitude. This creates severe inefficiency in naive static batching and motivates
the family of techniques covered here.

---

## Tier 1 — Fundamentals

### Q1. What is static batching and why does it waste GPU capacity for LLM inference?

**Answer.**

In static (or "batch-wait") batching, the server groups $B$ requests together,
runs a full forward pass for each generated token, and releases all requests
only when the **longest sequence in the batch** has finished generating.

**The core problem.** Suppose a batch contains one request that generates 2,048
tokens and seven requests that each generate 50 tokens. The seven short requests
finish at step 50, but their GPU memory and compute slots remain allocated until
step 2,048. The GPU is executing wasteful padding for those 7 sequences for
$2048 - 50 = 1998$ steps — roughly 97% of the batch's lifetime.

**Quantified waste.** If requests arrive with an exponential length distribution
(mean output length $\mu = 200$, max $= 2048$), the fraction of GPU compute
spent on padding is approximately:

$$
\text{waste} = 1 - \frac{\bar{L}}{L_\text{max}} = 1 - \frac{200}{2048} \approx 90\%
$$

This is not a theoretical worst case — it is representative of real LLM serving
workloads where some requests are short-answer queries and others are long
document completions within the same batch.

**Secondary problem.** Static batching also incurs a queuing latency: the server
must wait to accumulate $B$ requests before processing begins, increasing
time-to-first-token (TTFT) for requests that arrive right after a batch departs.

---

### Q2. What is continuous batching (also called iteration-level scheduling) and how does it solve the static batching problem?

**Answer.**

Continuous batching, introduced in the Orca system (Yu et al., 2022), changes
the granularity of batching from **request-level** to **iteration-level**. Rather
than holding a batch together for its entire lifetime, the server inspects the
batch after every single decode step and:

1. **Removes** any sequences that have just generated an EOS token (finished).
2. **Inserts** new sequences from the waiting queue into the freed slots.

The batch size fluctuates dynamically; slots are never left idle.

**Concrete example.** At step $t$:
- Sequence A finishes at token 42, freeing slot 3.
- Sequence D (from the queue) immediately enters slot 3 with its prefill.
- Sequences B, C, E–H continue their decode step as normal.
- The GPU runs a single fused forward pass over all active sequences.

**The key insight** that makes this feasible: during the decode phase, each
active sequence contributes exactly one new token per step. The attention
computation reuses the KV cache for all previous tokens (no recomputation needed).
Inserting a new sequence requires running its prefill (processing all its prompt
tokens) and then joining the decode batch.

---

### Q3. What is the KV cache and why is it essential for autoregressive inference?

**Answer.**

The KV cache stores the key and value tensors computed at each attention layer
for every previously generated token. During autoregressive generation, step $t$
generates token $x_t$ using attention over all tokens $x_1, \ldots, x_{t-1}$.
Without a cache, these key/value projections must be recomputed from scratch at
every step — an $O(T^2)$ total cost for a sequence of length $T$.

With the KV cache:
- Keys and values for positions $1, \ldots, t-1$ are read from cache.
- Only the key and value for the new token $x_t$ need to be computed.
- One matrix-vector product (not a matrix-matrix product) per head per step.

**Memory cost.** For a model with $L$ layers, $H$ attention heads, and head
dimension $d_h$, the KV cache for a single sequence at position $T$ is:

$$
\text{KV memory} = 2 \times L \times H \times d_h \times T \times \text{bytes per element}
$$

The factor 2 accounts for keys and values separately.

---

### Q4. What is PagedAttention and what problem does it solve?

**Answer.**

PagedAttention (Kwon et al., 2023, vLLM) applies the virtual memory paging
concept from operating systems to KV cache management.

**The problem it solves.** Without paging, each request's KV cache must be stored
in a contiguous block of GPU memory (because GPU CUDA kernels index memory with
pointer arithmetic). Memory must be pre-allocated for the maximum possible
sequence length, even if the actual generated length is much shorter. This
causes:

1. **Internal fragmentation.** A request that generates 100 tokens in a slot
   pre-allocated for 2,048 tokens wastes 1,948 token slots.
2. **Inability to share memory across requests.** System prompts (which are
   identical across requests in many serving scenarios) are duplicated in every
   request's KV cache.

**PagedAttention solution.** The KV cache is divided into fixed-size **pages**
(blocks of, e.g., 16 tokens). Each request is assigned pages on demand from a
global free-page pool. A **block table** per request maps logical page indices
to physical GPU memory addresses.

Benefits:
- Near-zero internal fragmentation (waste $\leq$ one partial page per request).
- **Copy-on-write prefix sharing:** multiple requests with the same system prompt
  share the same physical pages for those tokens until they diverge.
- The free-page pool allows the system to maximally fill GPU memory with active
  sequences, dramatically increasing throughput.

---

## Tier 2 — Intermediate

### Q5. Explain prefill-decode disaggregation. What is the motivation and what are the engineering challenges?

**Answer.**

**Motivation.** The prefill phase (processing the user's prompt) and the decode
phase (generating output tokens one at a time) have fundamentally different
computational characteristics:

| Property | Prefill | Decode |
|----------|---------|--------|
| Operation type | Matrix-matrix (GEMM) | Matrix-vector (GEMV) |
| Arithmetic intensity | High (compute-bound) | Low (memory-bandwidth-bound) |
| Latency | Proportional to prompt length | Proportional to output length |
| GPU utilisation | Near 100% | Often 20–40% |
| Optimal batch size | Small (each prompt is parallelised internally) | Large (many concurrent sequences) |

Running prefill and decode on the same GPU forces sub-optimal hardware utilisation
for both: prefill is interrupted by ongoing decode requests, and decode GPU
utilisation is limited.

**Disaggregation.** Separate the system into:
- **Prefill servers** (P-servers): dedicated to processing prompts. Can saturate
  GPU compute because the prompt tokens are all processed in parallel.
- **Decode servers** (D-servers): dedicated to generating tokens. Can maintain
  high memory-bandwidth utilisation with large batches of concurrent decode
  sequences.

When a prompt arrives, a P-server processes it, computes the KV cache, and
transfers that KV cache to a D-server, which then handles all subsequent token
generation.

**Engineering challenges.**

1. **KV cache transfer bandwidth.** A KV cache for a 2K-token prompt in LLaMA-2
   7B occupies approximately 1 GB (see worked problem 01). Transferring this
   over PCIe (16 GB/s) takes ~62ms — potentially comparable to the prefill time
   itself. NVLink or InfiniBand mitigates this.

2. **Load balancing.** The ratio of prefill to decode compute varies with workload.
   A chatbot workload (short prompts, short answers) requires fewer P-servers
   than a document summarisation workload (long prompts).

3. **Increased system complexity.** Failure in the KV cache transfer pipeline
   stalls decode; the system needs retry and fallback logic.

---

### Q6. What is chunked prefill and why does it improve time-to-first-token?

**Answer.**

**The problem.** A long prompt (e.g., 8,192 tokens in a RAG context) occupies
the GPU for tens or hundreds of milliseconds during prefill. All decode-phase
requests in the same batch are blocked waiting for this prefill to complete,
degrading their per-token latency.

**Chunked prefill** (Agrawal et al., 2023) breaks the prefill of each new request
into fixed-size **chunks** (e.g., 512 tokens) and interleaves them with decode
iterations:

- Iteration 1: process first 512 tokens of the new prompt (chunk 0) + decode step
  for all current requests.
- Iteration 2: process tokens 513–1024 of the prompt (chunk 1) + decode step.
- ...
- Iteration 16: process final chunk + decode step. Prefill is complete, request
  joins decode batch.

**Effect on latency.** Decode requests are delayed by at most one chunk worth of
compute per step rather than the full prefill. For a chunk size of 512 tokens,
the decode inter-token latency increase is bounded by the cost of processing
512 tokens — typically a few milliseconds.

**Effect on throughput.** Chunked prefill allows the GPU to work on both prefill
and decode simultaneously (they are batched together in a single forward pass),
improving utilisation compared to alternating full-prefill and full-decode passes.

**Trade-off.** Smaller chunks reduce latency impact on decode but increase
the number of forward passes (and thus overhead) required to complete a prefill.
Chunk size is tuned to balance decode latency degradation against prefill
throughput.

---

### Q7. How do scheduling policies (FCFS, preemption, priority) affect LLM serving quality of service?

**Answer.**

**First-Come-First-Served (FCFS).** Default policy: serve requests in arrival
order. Simple, but suffers from head-of-line blocking — a long, slow request
stalls shorter requests behind it. For interactive applications, P99 latency can
be very high.

**Shortest-Job-First (SJF).** Prioritise requests with the shortest *estimated*
output length. Minimises mean latency but requires output length prediction (which
is imprecise for open-ended generation) and can starve long requests.

**Preemptive scheduling.** If a running sequence is deprioritised (e.g., because
a higher-priority request arrives), its KV cache can be:
- **Swapped to CPU DRAM** (high-latency reload when resumed).
- **Recomputed from scratch** (wastes compute but avoids CPU-GPU transfer).
- **Discarded** (only viable if the request can tolerate being restarted entirely).

PagedAttention's page-based KV cache makes swapping more practical: only the
pages belonging to the preempted sequence need to be swapped, not a contiguous
block.

**Priority queues.** Enterprise deployments typically run multiple tiers:
- Tier 1 (interactive): latency SLA of <500ms TTFT, <50ms inter-token.
- Tier 2 (batch): throughput SLA, latency secondary.

The scheduler fills decode slots with Tier 1 first; Tier 2 requests fill remaining
capacity. Tier 1 can preempt Tier 2 when new Tier 1 requests arrive.

**Practical consideration.** vLLM's default scheduler uses a combination of FCFS
with preemption via swap-to-CPU. SGLang and TensorRT-LLM offer priority-based
scheduling with configurable policies.

---

## Tier 3 — Advanced

### Q8. Derive the throughput and latency trade-off in continuous batching. Under what conditions does increasing batch size stop improving throughput?

**Answer.**

**Model.** In the memory-bandwidth-bound decode regime, throughput (tokens per
second) scales with batch size up to the point where:

1. **GPU memory is exhausted** by KV caches (hard limit).
2. **Compute becomes the bottleneck** instead of memory bandwidth.

**Memory-bandwidth bound analysis.** Each decode step requires loading all model
parameters from HBM once (for the single new token per sequence). The time for
one decode step is:

$$
t_\text{decode} \approx \frac{2N}{M_\text{bw}} + \frac{B \cdot S_\text{KV}}{M_\text{bw}}
$$

where $N$ is parameter count (bytes), $M_\text{bw}$ is memory bandwidth (bytes/s),
$B$ is batch size, and $S_\text{KV}$ is KV cache size per token per sequence.

The first term (parameter loading) is constant per step; the second term grows
with batch size as more KV cache is accessed. Throughput (tokens per second) is:

$$
\text{Throughput} = \frac{B}{t_\text{decode}} = \frac{B}{\frac{2N + B \cdot S_\text{KV} \cdot T}{M_\text{bw}}}
$$

where $T$ is the average sequence length in the batch.

At small $B$: throughput $\propto B$ (the parameter loading dominates, and
each additional sequence adds nearly free tokens).

At large $B$: KV cache access $B \cdot S_\text{KV} \cdot T$ becomes comparable
to parameter loading $2N$, and throughput saturates.

**The cross-over batch size.** Throughput growth from increasing $B$ slows when:

$$
B \cdot S_\text{KV} \cdot T \approx 2N
\implies B_\text{sat} \approx \frac{2N}{S_\text{KV} \cdot T}
$$

For LLaMA-2 7B in BF16:
- $N = 7 \times 10^9 \times 2 = 14$ GB
- $S_\text{KV}$ per token per sequence $= 2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes $\approx 0.5$ MB per token
- At $T = 512$: $S_\text{KV} \cdot T = 256$ MB per sequence

$$
B_\text{sat} \approx \frac{14 \times 10^3 \text{ MB}}{256 \text{ MB}} \approx 55 \text{ sequences}
$$

Beyond $B \approx 55$ concurrent sequences, throughput gains diminish rapidly
for 512-token average sequences. This aligns well with empirical observations.

---

### Q9. What is the prefill-decode compute imbalance and how does it affect TTFT scaling with prompt length?

**Answer.**

**Prefill compute.** Processing a prompt of $P$ tokens requires computing
attention over all $\binom{P}{2}$ token pairs (the lower-triangular attention
mask). The total attention FLOPs for prefill scale as $O(P^2 d_\text{model})$.
Feed-forward layers scale as $O(P d_\text{FFN})$. For typical models where
$d_\text{FFN} \gg d_\text{model}^2 / L$ (i.e., for short-to-medium prompts),
the FFN dominates and TTFT $\propto P$. For very long prompts, the attention
quadratic term dominates and TTFT $\propto P^2$.

**Practical breakeven.** For LLaMA-2 7B with hidden size 4096 and FFN dimension
11008:

Attention FLOPs per layer $\approx 4 P^2 d_\text{model}$

FFN FLOPs per layer $\approx 4 P d_\text{FFN}$

Attention dominates when $P > d_\text{FFN} / d_\text{model} = 11008 / 4096 \approx 2.7$.

Wait — this cannot be right for typical prompts. Let us restate more carefully:
attention at layer $l$ for a prompt of length $P$ costs $4 P^2 d_h H$ FLOPs
(for QK, softmax, AV), while FFN costs $2 P \cdot 4 d_\text{model}^2$ FLOPs.
Attention dominates when:

$$
4 P^2 d_h H > 8 P d_\text{model}^2
\implies P > \frac{2 d_\text{model}^2}{d_h H} = \frac{2 d_\text{model}}{1} = 2 d_\text{model}
$$

(using $d_h H = d_\text{model}$). For LLaMA-2 7B with $d_\text{model} = 4096$:

$$
P_\text{breakeven} \approx 2 \times 4096 = 8192 \text{ tokens}
$$

So for prompts under 8K tokens, TTFT scales approximately linearly. Above 8K
tokens, the quadratic attention term begins to dominate and TTFT scales
super-linearly. This is the regime where FlashAttention and ring attention
become critically important for serving latency.

---

### Q10. Describe the Orca scheduling algorithm in detail. How does it handle heterogeneous sequence lengths within a batch?

**Answer.**

**Orca's core mechanism: selective batching.** The Orca paper (Yu et al., 2022)
observes that not all operations in a Transformer forward pass need to be batched
identically. It distinguishes:

1. **Attention layers:** sequences in the same batch can have different KV cache
   lengths. Attention is inherently per-sequence (each sequence attends to its
   own KV cache). These are computed independently and do not require padding.

2. **Feed-forward layers:** operations are per-token position, not per-sequence.
   Tokens from all sequences (at their respective positions) can be concatenated
   and processed in a single matrix multiplication (the "token-batching" insight).

**Iteration-level scheduling.** At each decode step, Orca:

1. Collects all sequences in the **running set** (currently generating).
2. For any sequence that completed in the previous step (EOS token emitted),
   removes it from the running set and moves the next waiting request into
   the **pending-join** list.
3. Runs the **prefill** for pending-join sequences (as a separate pass, or
   chunked into the current step).
4. Executes a single combined forward pass: FFN layers process all tokens from
   all sequences as a flat batch (no padding), attention layers process each
   sequence independently using its KV cache.
5. Returns generated tokens to clients.

**Handling heterogeneous lengths.** The key data structure is a **flat token
buffer**: instead of a $(B, T_\text{max}, d)$ tensor with padding, Orca uses a
$(T_\text{total}, d)$ tensor where $T_\text{total} = \sum_i T_i$ is the total
number of active tokens. A **sequence index mapping** tracks which tokens belong
to which sequence. This eliminates padding FLOPs entirely.

For attention, which must respect per-sequence causality, the variable-length
attention is dispatched with a per-sequence offset array. FlashAttention-2 and
later kernels natively support variable-length batched attention
(`flash_attn_varlen_func` in the FlashAttention library).

**Performance gain.** Orca reported $10\times$ to $23\times$ improvement in
throughput over static batching on realistic serving workloads. The gains are
largest when the output length distribution has high variance (which is typical
of open-ended generation).
