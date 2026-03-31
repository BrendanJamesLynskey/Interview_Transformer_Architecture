# Quiz: Inference Optimisation

**18 multiple-choice questions** covering KV cache memory, quantisation (INT4 / INT8),
LoRA/adapter methods, speculative decoding, continuous batching, and PagedAttention.

Difficulty: Intermediate through Advanced.

---

## Questions

---

### Q1 — KV cache memory formula

For a model with $L$ layers, $H$ attention heads, head dimension $d_h$, batch size $B$, and
sequence length $T$, what is the KV cache size in bytes when stored in FP16?

**A.** $L \cdot H \cdot d_h \cdot T \cdot B \cdot 2$

**B.** $2 \cdot L \cdot B \cdot H \cdot T \cdot d_h \cdot 2$

**C.** $L \cdot B \cdot d_{\text{model}} \cdot T \cdot 2$

**D.** $4 \cdot L \cdot H \cdot d_h \cdot T$

---

### Q2 — KV cache memory at scale

A LLaMA-2-70B model has 80 layers, 8 KV heads (GQA), head dimension 128, and generates sequences of
up to 4096 tokens.  Approximately how large is the KV cache per request in FP16?

**A.** ~500 MB

**B.** ~160 MB

**C.** ~80 MB

**D.** ~5 GB

---

### Q3 — INT8 post-training quantisation (PTQ)

In weight-only INT8 quantisation for inference, model weights are stored as INT8 but
activations remain in FP16.  What is the primary benefit and primary risk?

**A.** Benefit: 2x memory reduction for weights; risk: any arithmetic error is catastrophic
because INT8 is not representable in FP16.

**B.** Benefit: ~2x reduction in weight memory footprint, enabling larger models to fit on a
single GPU; risk: quantisation error, particularly for outlier activation channels, can
degrade model quality, requiring calibration or mixed-precision schemes.

**C.** Benefit: 4x memory reduction; risk: the model can no longer be fine-tuned.

**D.** Benefit: eliminates the need for KV cache; risk: slower prefill.

---

### Q4 — INT4 vs INT8 quantisation

Compared to INT8, what additional challenge does INT4 weight quantisation introduce?

**A.** INT4 quantisation requires the model to be retrained from scratch.

**B.** INT4 uses only 16 representable values per weight, making it much more sensitive to
quantisation error.  Effective INT4 methods (e.g., GPTQ, AWQ) use group-wise quantisation
(separate scale/zero-point per group of 128 weights) and sometimes compensate high-salience
weights, trading additional bookkeeping for acceptable accuracy loss.

**C.** INT4 cannot be implemented on NVIDIA GPUs due to hardware limitations.

**D.** INT4 quantisation always produces worse perplexity than INT8 by more than 10 PPL points
regardless of calibration.

---

### Q5 — Activation quantisation challenges

LLM.int8() (Dettmers et al., 2022) identified "emergent outliers" in LLM activations.  What
are these and why do they complicate quantisation?

**A.** Outlier activations are tokens in the input that are longer than 512 characters.

**B.** In large LLMs (typically beyond ~6.7B parameters), a small fraction of hidden dimensions
develop activation magnitudes many times larger (100x or more) than typical.  Quantising these
dimensions together with normal ones causes massive rounding error.  LLM.int8() addresses this
by processing outlier dimensions in FP16 and quantising the remainder in INT8 (mixed-precision
decomposition).

**C.** Outlier activations are NaN values that appear in the attention softmax for very long
sequences.

**D.** Outlier activations only occur in the embedding layer and do not affect quantisation of
attention or FFN layers.

---

### Q6 — LoRA: rank and parameter count

LoRA (Hu et al., 2021) adapts a frozen pretrained weight $W \in \mathbb{R}^{m \times n}$ by
learning a low-rank update $\Delta W = BA$ where $B \in \mathbb{R}^{m \times r}$ and
$A \in \mathbb{R}^{r \times n}$.  If $m = n = 4096$ and $r = 16$, how many trainable parameters
does LoRA introduce per weight matrix (ignoring any scaling constant)?

**A.** $4096^2 = 16{,}777{,}216$

**B.** $2 \times 4096 \times 16 = 131{,}072$

**C.** $16^2 = 256$

**D.** $4096 \times 16 = 65{,}536$

---

### Q7 — LoRA rank selection

How does the choice of LoRA rank $r$ affect the adapter's expressiveness and parameter count?

**A.** Higher $r$ allows a higher-rank update to $W$, capturing more complex adaptations, at the
cost of more trainable parameters (linear in $r$) and slightly higher risk of overfitting on
small datasets.

**B.** Higher $r$ reduces overfitting because the adapter matrix becomes more regularised.

**C.** The rank $r$ only affects initialisation and has no impact on the expressiveness of the
update.

**D.** Ranks above $r = 4$ are never useful; typical fine-tuning saturates at $r = 4$.

---

### Q8 — Speculative decoding -- mechanism

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates large-model
inference.  Which description is correct?

**A.** A small "draft" model generates $k$ candidate tokens autoregressively; the large "target"
model verifies all $k$ tokens in a single parallel forward pass; tokens are accepted or rejected
based on the ratio of target to draft probabilities; the first rejected token is resampled from
the corrected distribution.

**B.** The large model generates tokens in parallel by running $k$ independent forward passes
simultaneously on separate GPUs.

**C.** The draft model provides token embeddings that skip the target model's embedding layer.

**D.** The target model is distilled into the draft model during inference.

---

### Q9 — Speculative decoding acceptance probability

In speculative decoding, token $x$ proposed by the draft model is accepted with probability:

$$\alpha = \min\!\left(1,\; \frac{p_{\text{target}}(x \mid \text{context})}{p_{\text{draft}}(x \mid \text{context})}\right)$$

Under what condition is every proposed token accepted?

**A.** When $p_{\text{draft}}(x) = 0$ for all tokens.

**B.** When the draft model's distribution matches the target model's distribution exactly:
$p_{\text{draft}} = p_{\text{target}}$.  In this case $\alpha = 1$ everywhere, and the
expected number of tokens accepted per step equals $k$.

**C.** When the target model is more confident than the draft model on every token.

**D.** Acceptance is always probabilistic; 100% acceptance is theoretically impossible.

---

### Q10 — Continuous batching

Naively, a serving system processes each request as a separate batch or pads requests in a
static batch to the same length.  What problem does continuous batching (Orca, Yu et al., 2022)
solve?

**A.** It prevents the KV cache from exceeding GPU memory by evicting long sequences.

**B.** Static batching holds the GPU idle while waiting for the longest request in a batch to
finish.  Continuous batching interleaves the decode steps of different requests: when any
request finishes, a new request immediately fills its slot, keeping GPU utilisation near 100%
and dramatically improving throughput.

**C.** It batches gradient updates from multiple fine-tuning jobs running simultaneously.

**D.** It compresses the KV cache across requests that share a common prefix.

---

### Q11 — PagedAttention

PagedAttention (Kwon et al., 2023, vLLM) is inspired by virtual memory paging in operating
systems.  What problem does it address?

**A.** It speeds up the prefill phase by paging token embeddings from disk.

**B.** Naively, KV cache must be pre-allocated as a contiguous block for each request's maximum
sequence length, wasting memory when actual sequences are shorter.  PagedAttention divides the
KV cache into fixed-size pages (blocks), allocated non-contiguously on demand, and a block table
maps each request's logical sequence positions to physical memory pages.  This eliminates
internal fragmentation and allows memory sharing between parallel sampling sequences.

**C.** It pages the model weights to CPU RAM during decoding to fit larger models.

**D.** It replaces the KV cache with a hash-based lookup table to deduplicate cached keys.

---

### Q12 — KV cache quantisation

KV cache entries can be quantised to INT8 or INT4 to reduce memory pressure.  Why is KV cache
quantisation distinct from weight quantisation?

**A.** KV cache tensors are read once and discarded, so quantisation error accumulates over only
one forward pass.

**B.** KV tensors are written during prefill and read at every subsequent decode step; quantisation
error is reintroduced at each read and can accumulate over long sequences, affecting output
quality differently from static weight quantisation.  Additionally, KV tensors have different
statistical properties (e.g., key activations may have outliers on certain heads) requiring
per-head or per-token quantisation granularity.

**C.** KV cache quantisation is straightforward because keys and values have uniform distributions.

**D.** KV cache quantisation reduces the number of attention heads automatically.

---

### Q13 — Prefill vs decode phases

Autoregressive LLM inference has two distinct computational phases.  Which statement correctly
characterises the difference?

**A.** Prefill is memory-bandwidth-bound; decode is compute-bound.

**B.** Prefill processes the input prompt in parallel (all tokens at once), is compute-bound
(high arithmetic intensity), and builds the initial KV cache.  Decode generates tokens one at a
time using the KV cache; it is memory-bandwidth-bound because each step reads the entire KV
cache with low reuse ratio.

**C.** Prefill and decode have identical computational profiles; the difference is only in the
number of tokens processed.

**D.** Prefill is performed on CPU; decode is performed on GPU.

---

### Q14 — Tensor parallelism for inference

Tensor parallelism (e.g., Megatron-LM style) splits attention heads and FFN weight matrices
across GPUs.  What is its primary purpose in inference?

**A.** It reduces the total number of arithmetic operations required.

**B.** It allows model shards that exceed a single GPU's memory capacity to be served by
distributing the weight matrices, enabling inference of 70B+ parameter models across multiple
GPUs with near-linear memory scaling.

**C.** It eliminates the need for KV cache by distributing sequence positions across GPUs.

**D.** It provides fault tolerance; if one GPU fails, the other GPUs continue inference.

---

### Q15 — GPTQ quantisation algorithm

GPTQ (Frantar et al., 2022) is a one-shot post-training quantisation method for LLMs.  What
distinguishes it from simple round-to-nearest (RTN) quantisation?

**A.** GPTQ uses gradient descent to adjust weights after quantisation.

**B.** GPTQ uses second-order (Hessian-based) information derived from a small calibration set
to find quantised weights that minimise the layer-wise reconstruction error.  It updates
remaining (not-yet-quantised) weights to compensate for the error introduced by quantising
each weight, producing significantly better accuracy than RTN at low bit-widths.

**C.** GPTQ requires full fine-tuning of the model on the target domain after quantisation.

**D.** GPTQ quantises activations as well as weights, unlike RTN which quantises only weights.

---

### Q16 — Chunked prefill

Chunked prefill is a technique used in high-throughput inference systems.  What does it do?

**A.** It splits the input vocabulary into chunks processed by separate GPUs.

**B.** It splits the prompt into smaller chunks processed in multiple forward passes, allowing
decode steps for already-started requests to be interleaved with prefill steps for new requests.
This reduces the "time to first token" stall that long prompts cause in continuous batching.

**C.** It pre-compresses the prompt using a summarisation model before prefill.

**D.** It skips attention computation for prompt tokens that have low attention entropy.

---

### Q17 — Prompt caching (prefix caching)

Some serving systems (e.g., vLLM with prefix caching, Anthropic's prompt caching) cache the
KV states of a shared prompt prefix.  What is the benefit and the constraint?

**A.** Benefit: eliminates all prefill computation for any request; constraint: requires a fixed
batch size.

**B.** Benefit: when multiple requests share a common prefix (e.g., a system prompt), the KV
cache for that prefix is computed once and reused, reducing prefill latency and compute cost
for subsequent requests; constraint: the prefix must be identical across requests and the cache
must fit in GPU memory.

**C.** Benefit: reduces model parameter count by merging shared prefix representations into
weights; constraint: the prefix cannot exceed 128 tokens.

**D.** Benefit: allows streaming generation to start before the full prompt is received;
constraint: the model must support bidirectional attention.

---

### Q18 — Speculative decoding speedup conditions

Speculative decoding achieves the best wall-clock speedup under which conditions?

**A.** When the draft and target models have the same parameter count.

**B.** When the draft model is much smaller and faster than the target model, the target model's
forward pass over $k$ tokens in parallel is not much more expensive than a single-token pass
(true at small batch sizes where decoding is memory-bound), and the draft model's distribution
is close enough to the target's that acceptance rates are high (typically > 70%).

**C.** When the draft model produces tokens with the highest possible perplexity.

**D.** When GPU memory is fully utilised by the target model, leaving no room for the draft model.

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | B      |
| 2  | B      |
| 3  | B      |
| 4  | B      |
| 5  | B      |
| 6  | B      |
| 7  | A      |
| 8  | A      |
| 9  | B      |
| 10 | B      |
| 11 | B      |
| 12 | B      |
| 13 | B      |
| 14 | B      |
| 15 | B      |
| 16 | B      |
| 17 | B      |
| 18 | B      |

---

## Detailed Explanations

---

### Q1 — KV cache memory formula

**Correct: B.**

The formula is: $2 \times L \times B \times H \times T \times d_h \times \text{bytes\_per\_element}$.

The factor of 2 accounts for both K and V tensors.  FP16 uses 2 bytes per element.
So: $2 \cdot L \cdot B \cdot H \cdot T \cdot d_h \cdot 2$.

- **A is wrong.** Missing the factor of 2 for both K and V, and the 2-byte FP16 factor is
  misapplied.
- **C is wrong.** $d_{\text{model}}$ conflates heads and head dimension; the correct form
  uses $H \cdot d_h = d_{\text{model}}$ but the formula structure is wrong.
- **D is wrong.** Missing the batch dimension and has incorrect byte counting.

---

### Q2 — KV cache at scale

**Correct: B.**

LLaMA-2-70B uses GQA with 8 KV heads. Using the KV cache formula:

$\text{KV cache} = 2 \times L \times B \times H_{KV} \times T \times d_h \times \text{bytes\_per\_element}$

$= 2 \times 80 \times 1 \times 8 \times 4096 \times 128 \times 2 \approx 167$ MB.

This is approximately 160 MB, matching option B.

- **A is wrong.** 500 MB would correspond to an intermediate head count.
- **C is wrong.** 80 MB would require even fewer KV heads or shorter sequences.
- **D is wrong.** 5 GB would correspond to the full MHA (64 K/V heads) case without GQA.

---

### Q3 — INT8 PTQ

**Correct: B.**

Weight-only INT8 quantisation halves the weight memory (from 2 bytes FP16 to 1 byte INT8).
For a 70B-parameter model this reduces weight storage from ~140 GB to ~70 GB, potentially
enabling single-node 8×80GB deployment.  The main risk is outlier activation channels: a few
hidden dimensions have much larger dynamic range, and quantising their corresponding weight rows
with the same scale as typical rows introduces large relative error.

- **A is wrong.** INT8 and FP16 are both representable number formats; there is no fundamental
  incompatibility, only precision loss.
- **C is wrong.** Weight-only INT8 gives ~2x memory reduction per weight, not 4x (which would
  be INT4).
- **D is wrong.** Weight quantisation does not affect the KV cache at all; the KV cache stores
  activation tensors.

---

### Q4 — INT4 vs INT8

**Correct: B.**

INT4 has values in $\{-8, \ldots, 7\}$ (signed) or $\{0, \ldots, 15\}$ (unsigned) -- only 16
distinct levels.  Even with calibration, representing the full dynamic range of FP16 weights in
4 bits requires careful group-wise scaling.  Methods like GPTQ, AWQ, and QuIP# use different
strategies (Hessian-based error compensation, activation-aware scaling, incoherence processing)
to achieve near-FP16 quality at INT4.

- **A is wrong.** PTQ methods like GPTQ quantise a pretrained model without retraining; only
  QAT (Quantisation-Aware Training) requires training.
- **C is wrong.** NVIDIA GPUs support INT4 via WMMA (Warp-level Matrix Multiply Accumulate)
  instructions on Turing and later architectures.
- **D is wrong.** Well-calibrated INT4 methods (e.g., GPTQ on LLaMA-2-7B) typically show
  perplexity increases of 0.1--0.5 PPL, not 10+ PPL.

---

### Q5 — LLM.int8() outlier activations

**Correct: B.**

Dettmers et al. found that starting around 6.7B parameters, a few feature dimensions (sometimes
fewer than 10 out of 4096) develop magnitudes up to 60,000x larger than the median.  These
outliers appear consistently across tokens and layers once they emerge.  Treating them with the
same INT8 scale as normal dimensions forces either clipping (losing the outlier) or a very large
scale (losing precision for normal values).  The mixed-precision decomposition decomposes the
matrix product, handling outlier columns in FP16 and the remainder in INT8.

- **A is wrong.** Outliers refer to activation tensor values, not input token lengths.
- **C is wrong.** NaN values are a separate numerical stability issue; outlier activations are
  large but finite.
- **D is wrong.** Outliers appear throughout the network in attention and FFN layers; they are
  not confined to the embedding layer.

---

### Q6 — LoRA parameter count

**Correct: B.**

LoRA introduces two matrices: $B \in \mathbb{R}^{4096 \times 16}$ and $A \in \mathbb{R}^{16 \times 4096}$.
Parameter counts: $4096 \times 16 + 16 \times 4096 = 65{,}536 + 65{,}536 = 131{,}072$.
This is ~128x fewer parameters than the full weight matrix ($4096^2 = 16{,}777{,}216$).

- **A is wrong.** That is the full weight matrix parameter count; LoRA is designed to avoid
  updating this.
- **C is wrong.** $16^2 = 256$ would be the count if BOTH matrices were $r \times r$, which they
  are not.
- **D is wrong.** $4096 \times 16 = 65{,}536$ is the count for one matrix only (A or B), not both.

---

### Q7 — LoRA rank selection

**Correct: A.**

Rank $r$ is the intrinsic dimension of the weight update.  Higher $r$ allows $\Delta W$ to span
a larger subspace, capturing more complex task-specific adaptations.  The parameter count of the
LoRA adapter scales as $2 \cdot r \cdot (m + n) / 2 \approx r \cdot d_{\text{model}}$, growing
linearly with $r$.  For small fine-tuning datasets, high-rank adapters may overfit.

- **B is wrong.** Higher rank means more parameters and LESS regularisation, not more.
- **C is wrong.** Rank directly controls the expressiveness of the update; $r = 1$ allows only
  a rank-1 update, $r = 16$ allows a rank-16 update, which is far more expressive.
- **D is wrong.** Optimal rank is task- and dataset-dependent.  Coding tasks or multilingual
  adaptation often benefit from $r = 64$ or higher; simple style adaptation may saturate at low
  rank.

---

### Q8 — Speculative decoding mechanism

**Correct: A.**

The algorithm:
1. Draft model generates tokens $x_1, \ldots, x_k$ autoregressively.
2. Target model runs ONE forward pass with the prompt + all $k$ draft tokens as input,
   obtaining target probabilities $p_{\text{target}}(x_t \mid \text{prefix})$ for each position.
3. For each position $t$: accept $x_t$ with probability $\min(1, p_T / p_D)$.
4. On rejection at position $t$: sample a corrected token from
   $\text{norm}(\max(0, p_T - p_D))$ and discard tokens $x_{t+1}, \ldots, x_k$.

This guarantees the output distribution exactly matches the target model.

- **B is wrong.** The target model is not run in parallel; it runs once over all $k$ tokens
  together in a single forward pass.
- **C is wrong.** Draft tokens go through the full target model including embedding layers.
- **D is wrong.** The models remain separate; there is no online distillation.

---

### Q9 — Acceptance probability condition

**Correct: B.**

When $p_{\text{draft}} = p_{\text{target}}$, we have $p_T / p_D = 1$ everywhere, so
$\alpha = \min(1, 1) = 1$ for all tokens.  Every token is accepted.  Conversely, when the
draft distribution diverges from the target, acceptance rates drop, and on average fewer tokens
are accepted per step.

- **A is wrong.** If $p_{\text{draft}}(x) = 0$, the draft model would never propose $x$; this
  condition is about tokens that ARE proposed.
- **C is wrong.** If the target is more confident ($p_T > p_D$ for the proposed token),
  acceptance is higher for that token, but other tokens where $p_T < p_D$ would still be
  rejected.  This is not a condition for 100% acceptance.
- **D is wrong.** If the draft matches the target, the algorithm does achieve 100% acceptance
  in expectation; it is theoretically achievable.

---

### Q10 — Continuous batching

**Correct: B.**

In static batching, a batch of $n$ requests finishes only when the last request completes.
Short requests wait idly while the long ones continue, wasting GPU cycles.  Continuous batching
(also called "iteration-level scheduling") swaps out finished requests and inserts new ones at
the granularity of a single decode iteration, achieving close to 100% GPU utilisation.  This
gives 3--10x throughput improvements over static batching in practice.

- **A is wrong.** That describes KV cache eviction policies (e.g., sliding window in Orca), not
  continuous batching itself.
- **C is wrong.** Continuous batching is an inference serving technique, not a training gradient
  accumulation method.
- **D is wrong.** Shared prefix compression is a separate optimisation (prompt/prefix caching).

---

### Q11 — PagedAttention

**Correct: B.**

In pre-PagedAttention systems, each request's KV cache must be allocated as a contiguous region
large enough for the maximum sequence length.  Sequences shorter than this maximum leave memory
unused (internal fragmentation).  PagedAttention (inspired by OS virtual memory) allocates
fixed-size "pages" of KV cache on demand, allowing memory sharing between beam-search
hypotheses (which share a common prefix page) and near-zero waste.  This roughly doubles
effective GPU memory utilisation for KV caches.

- **A is wrong.** PagedAttention manages GPU memory; it does not involve disk I/O during
  inference.
- **C is wrong.** Model weight offloading to CPU RAM is a different technique (e.g., used by
  DeepSpeed ZeRO-Infinity).
- **D is wrong.** Hash-based deduplication is a separate idea; PagedAttention is about memory
  allocation layout, not content-based deduplication.

---

### Q12 — KV cache quantisation specifics

**Correct: B.**

Static weight quantisation introduces a fixed, one-time error.  KV cache tensors are dynamic:
keys and values are computed freshly for each sequence and read repeatedly (once per decode
step) for the entire lifetime of the request.  Quantisation error in the KV cache can cause
attention scores to be systematically biased, and this bias accumulates as the sequence grows.
Keys in particular often have highly non-uniform distributions (certain heads exhibit outliers
correlated with semantic content), requiring per-head or per-token granularity for acceptable
accuracy.

- **A is wrong.** KV cache entries are READ at EVERY decode step after they are written, not
  just once.
- **C is wrong.** KV activations are not uniformly distributed; outlier heads are well-documented.
- **D is wrong.** KV cache quantisation reduces memory consumption, not the number of attention
  heads.

---

### Q13 — Prefill vs decode phases

**Correct: B.**

Prefill has high arithmetic intensity (many tokens processed in parallel, reusing loaded weights
across the full batch of tokens), making it compute-bound.  Decode processes one new token at a
time per request; the bottleneck is loading the KV cache and weight matrices from HBM for each
decode step.  Since the compute per loaded byte is very low, decode is memory-bandwidth-bound.
This distinction explains why prefill throughput scales with FLOPs (bigger GPUs help) while
decode throughput scales with memory bandwidth.

- **A is wrong.** This reverses the characterisation: prefill is compute-bound, decode is
  memory-bandwidth-bound.
- **C is wrong.** The computational profiles differ fundamentally (parallel vs sequential,
  high vs low arithmetic intensity).
- **D is wrong.** Both phases run on GPU in standard deployment; there is no CPU prefill stage.

---

### Q14 — Tensor parallelism for inference

**Correct: B.**

A 70B model in FP16 occupies ~140 GB.  A single A100 80GB GPU cannot hold the full model.
Tensor parallelism splits the weight matrices across $n$ GPUs, each holding $1/n$ of each
matrix.  The all-reduce communication cost is manageable over NVLink.  This is how vLLM,
TGI, and TensorRT-LLM serve large models in practice.

- **A is wrong.** Tensor parallelism does not reduce total FLOPs; it distributes them across
  devices.
- **C is wrong.** Tensor parallelism splits weight dimensions, not sequence positions (that would
  be sequence parallelism).
- **D is wrong.** Standard tensor parallelism has no fault tolerance; all GPUs must participate
  in the all-reduce operations at every layer.

---

### Q15 — GPTQ

**Correct: B.**

GPTQ builds on Optimal Brain Surgeon (OBS): it quantises one weight column at a time, computing
the second-order (Hessian) sensitivity to quantise in order of increasing error, and updates
the remaining unquantised weights to compensate.  This is done layer by layer using a
calibration set of ~128 sequences.  At 4-bit precision, GPTQ achieves near-FP16 perplexity
where simple RTN fails noticeably.

- **A is wrong.** GPTQ is a PTQ method; it uses the Hessian, not gradient descent, and does
  not require backward passes in the training sense.
- **C is wrong.** GPTQ is explicitly a post-training, one-shot method; no fine-tuning is needed.
- **D is wrong.** GPTQ is a weight-only quantisation method; activations remain in floating
  point during inference.

---

### Q16 — Chunked prefill

**Correct: B.**

Long prompts can monopolise the GPU for tens to hundreds of milliseconds during prefill,
stalling decode steps for other already-started requests and increasing their inter-token
latency.  Chunked prefill splits the prompt into, say, 512-token chunks.  The serving scheduler
interleaves these chunks with decode steps, ensuring decode latency stays bounded even when
simultaneously admitting long-prompt requests.

- **A is wrong.** Vocabulary chunking is unrelated to the prefill/decode pipeline.
- **C is wrong.** No summarisation is done; the full prompt is processed, just in smaller pieces.
- **D is wrong.** Chunked prefill does not skip attention computation; it schedules it in smaller
  time slices.

---

### Q17 — Prompt caching

**Correct: B.**

Anthropic's API prompt caching (and vLLM's prefix caching) reuses KV cache blocks for any
request that begins with an identical prefix to a previously processed request.  The prefix
must be byte-for-byte identical because the KV values depend on exact token sequences.  Memory
for cached prefixes must remain allocated between requests, so there is a memory footprint for
maintaining the cache.

- **A is wrong.** Only the CACHED prefix's prefill is skipped; the non-prefix portion must
  still be processed.
- **C is wrong.** The model weights remain unchanged; only activation (KV) tensors are cached.
- **D is wrong.** Prompt caching works with autoregressive (causal) models; bidirectional
  attention is not required.

---

### Q18 — Speculative decoding speedup conditions

**Correct: B.**

Three conditions jointly determine the speedup:
1. **Draft-target speed ratio**: if the target runs $k$ tokens only marginally more slowly than
   1 token (true when decoding is memory-bandwidth-bound and the batch is small), speculative
   decoding is efficient.
2. **Acceptance rate**: if the draft distribution closely approximates the target's, most
   proposed tokens are accepted, and the expected accepted tokens per step approaches $k$.
3. **Draft model size**: a smaller draft model (5--10x fewer parameters) generates the $k$ tokens
   much faster than the target would, making the overall pipeline faster.

- **A is wrong.** If draft and target have the same parameter count, using the draft model
  wastes compute (it is as expensive as the target itself).
- **C is wrong.** High draft perplexity implies the draft distribution diverges from the target,
  leading to low acceptance rates and poor speedup.
- **D is wrong.** If GPU memory is fully utilised by the target model, the draft model cannot
  reside on the same GPU, which is a practical constraint requiring careful system design
  (e.g., placing the draft on CPU or a separate GPU).
