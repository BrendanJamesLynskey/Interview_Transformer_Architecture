# Quiz: Modern Architecture Features

**18 multiple-choice questions** covering decoder-only LLMs, Rotary Positional Embeddings (RoPE),
Grouped-Query Attention (GQA), Mixture-of-Experts (MoE), FlashAttention, and RMSNorm vs LayerNorm.

Difficulty: Intermediate through Advanced.

---

## Questions

---

### Q1 — Decoder-only architecture

GPT-style models are "decoder-only" Transformers.  Which structural choice distinguishes them
from the original encoder-decoder Transformer?

**A.** They use only encoder blocks (bidirectional attention) and no decoder blocks.

**B.** They consist entirely of masked (causal) self-attention blocks with no cross-attention
component, making them suitable for autoregressive generation but not for tasks requiring a
separate encoder.

**C.** They replace self-attention with convolution layers for efficiency.

**D.** They omit positional encodings, relying on the attention pattern alone to infer order.

---

### Q2 — Rotary Positional Embeddings (RoPE) -- mechanism

RoPE (Su et al., 2021) encodes position by rotating the query and key vectors.  Which property
makes RoPE particularly appealing?

**A.** RoPE adds a learnable bias vector to each token embedding before the attention layer.

**B.** The dot product $q_m^\top k_n$ after rotation depends only on the content of $q$ and $k$
and their relative position $m - n$, not on their absolute positions.  This gives the model
a relative-position inductive bias while preserving compatibility with standard self-attention.

**C.** RoPE replaces the $\frac{1}{\sqrt{d_k}}$ scaling factor, reducing computation.

**D.** RoPE stores position information outside the attention computation in a separate memory
bank.

---

### Q3 — RoPE rotation formula

RoPE applies a block-diagonal rotation matrix $R_m$ to $d$-dimensional vectors using $d/2$
2D rotation sub-matrices, each with angle $m \cdot \theta_i$ where
$\theta_i = 10000^{-2i/d}$.  What does varying $\theta_i$ across dimensions achieve?

**A.** It prevents the model from attending to tokens at very large relative offsets.

**B.** Different dimensions rotate at different frequencies, from high-frequency (small $i$,
large $\theta_i$) to low-frequency (large $i$, small $\theta_i$), allowing the model to
distinguish both fine-grained and coarse-grained positional differences.

**C.** All dimensions rotate at the same angle; the index $i$ is used for normalisation only.

**D.** The varying $\theta_i$ is an approximation to sinusoidal positional encoding and produces
identical representations.

---

### Q4 — RoPE context length extension

LLaMA-2 models trained with a context window of 4096 tokens struggle when prompted with
sequences longer than this.  Which technique directly addresses this by adjusting RoPE
frequencies?

**A.** Increasing the number of attention heads.

**B.** Positional interpolation (Chen et al., 2023): linearly scale down the position indices so
that positions in the extended context map to values within the original training range,
effectively rescaling the rotation angles.

**C.** Using a larger learning rate during inference for the embedding layer.

**D.** Replacing RoPE with sinusoidal encodings for extended contexts.

---

### Q5 — Grouped-Query Attention (GQA) -- mechanism

GQA (Ainslie et al., 2023) is a generalisation of both Multi-Head Attention (MHA) and
Multi-Query Attention (MQA).  How does GQA differ from standard MHA?

**A.** GQA adds extra query heads beyond the number of key/value heads.

**B.** GQA partitions the $H$ query heads into $G$ groups ($G < H$), where all heads within a
group share a single set of key and value projection weights.  This reduces the KV parameter
count and KV cache size by a factor of $H / G$ while retaining more expressiveness than MQA
(which uses $G = 1$).

**C.** GQA replaces multi-head attention with a single global attention operation over the
full sequence.

**D.** GQA removes the value projection, computing outputs directly from the attention weights.

---

### Q6 — GQA memory savings

A model uses $H = 32$ query heads and $G = 8$ key/value head groups.  By what factor does GQA
reduce the size of the KV cache compared to standard MHA?

**A.** 2x

**B.** 4x

**C.** 8x

**D.** 32x

---

### Q7 — Mixture-of-Experts (MoE) -- sparsity

In a sparse MoE Transformer (e.g., Mixtral), the FFN sublayer is replaced with $E$ expert FFNs
and a router.  If the top-$k$ routing strategy selects $k = 2$ experts per token and there are
$E = 8$ experts total, what fraction of FFN parameters are used for any given token?

**A.** $1/8$ of the experts are active, so $1/8$ of FFN parameters are used.

**B.** $2/8 = 1/4$ of FFN parameters are used per token.

**C.** All 8 experts compute activations but only the top-2 outputs are weighted non-zero.

**D.** The router uses all experts; sparsity only applies to the attention layers.

---

### Q8 — MoE router training challenge

Training MoE models with learned routing faces a "collapse" problem.  What is this, and how is
it addressed?

**A.** The router weights collapse to zero, disabling all experts.  This is fixed by adding a
large positive bias.

**B.** The router learns to send all tokens to the same 1--2 experts (load imbalance), leaving
most experts undertrained.  An auxiliary load-balancing loss penalises uneven routing
distributions, encouraging utilisation of all experts.

**C.** Experts collapse to identical weights because they receive the same gradient.  This is
fixed by randomly initialising experts from different distributions.

**D.** The router cannot backpropagate through the hard top-$k$ selection, so gradients do not
reach the expert weights.

---

### Q9 — FlashAttention -- problem it solves

Standard attention materialises the full $T \times T$ attention matrix in GPU HBM (high-bandwidth
memory).  What problem does FlashAttention (Dao et al., 2022) address?

**A.** The $O(T^2 \cdot d)$ FLOPs of attention; FlashAttention reduces this to $O(T \log T \cdot d)$.

**B.** The $O(T^2)$ HBM memory read/write cost of storing and loading the attention matrix.
FlashAttention tiles the computation in SRAM (fast on-chip cache), never materialising the full
attention matrix, reducing HBM I/O from $O(T^2)$ to $O(T)$ and achieving significant
wall-clock speedups.

**C.** The quadratic parameter count of multi-head attention.

**D.** The inability of attention to handle sequences longer than the GPU's SRAM capacity.

---

### Q10 — FlashAttention -- correctness

FlashAttention avoids materialising the attention matrix but must still compute the softmax
correctly.  Which technique enables this?

**A.** It approximates the softmax using a Taylor expansion, accepting a small accuracy trade-off.

**B.** It uses the numerically stable online softmax update: when processing a new block of keys,
it rescales the previously accumulated output using the ratio of old and new normalisation
constants, producing an exact result without requiring the full row of scores at once.

**C.** It replaces softmax with a linear kernel approximation (random features), converting the
quadratic attention to linear.

**D.** It stores only the top-$k$ attention scores per row, dropping the rest before softmax.

---

### Q11 — RMSNorm vs LayerNorm

RMSNorm (Zhang & Sennrich, 2019) is used in LLaMA and many recent LLMs instead of standard
LayerNorm.  What is the key difference?

**A.** RMSNorm uses a batch statistic instead of a per-sample statistic.

**B.** RMSNorm normalises by the root mean square of the activations and omits the mean-centring
(re-centring) step of LayerNorm.  This reduces computation slightly and empirically performs
comparably while removing the bias parameter.

**C.** RMSNorm applies normalisation along the sequence dimension rather than the feature
dimension.

**D.** RMSNorm is equivalent to LayerNorm with a fixed scale parameter of 1.

---

### Q12 — SwiGLU activation

LLaMA and many decoder LLMs replace the FFN's GELU activation with SwiGLU.  What is SwiGLU?

**A.** $\text{SwiGLU}(x, W, V, W_2) = (x W \odot \text{Swish}(x V)) W_2$,
a gated linear unit where one linear branch gates another via element-wise multiplication,
with a Swish ($x \cdot \sigma(x)$) activation on the gate branch.

**B.** A second-order polynomial approximation to GELU that is faster to compute.

**C.** A normalised softmax applied to the hidden state of the FFN.

**D.** An acronym for "Switched Weight Linear Unit" which rotates FFN weights during training.

---

### Q13 — Sliding window attention (Longformer / Mistral)

Sliding window attention restricts each token to attend to a local window of $w$ tokens.  What
is the resulting complexity and trade-off?

**A.** $O(T)$ time and $O(w)$ memory; but the model loses all ability to model long-range
dependencies.

**B.** $O(T \cdot w)$ time and $O(T \cdot w)$ memory for the attention scores, a significant
reduction from $O(T^2)$ for long sequences.  Global tokens or alternating full-attention layers
can partially recover long-range modelling capability.

**C.** $O(T^2 / w^2)$ time, linearly better than full attention only when $w = \sqrt{T}$.

**D.** The complexity is the same as full attention because the window must be recalculated at
every token position.

---

### Q14 — KV head sharing across layers

Some architectures (e.g., cross-layer attention in certain LLM designs) share KV caches across
multiple consecutive Transformer layers.  What is the primary motivation?

**A.** Sharing KV weights across layers eliminates the need for residual connections.

**B.** It reduces KV cache memory at inference time and lowers parameter count by reusing the
same key/value projections for multiple layers, at the cost of potentially limiting each layer's
ability to attend to distinct information.

**C.** It speeds up training by reducing the number of backward passes required.

**D.** It prevents attention heads from attending to the same positions twice.

---

### Q15 — ALiBi positional encoding

ALiBi (Press et al., 2022) is an alternative positional scheme.  How does it encode position?

**A.** It adds a sinusoidal vector to each token embedding before the attention layer.

**B.** Instead of modifying embeddings, ALiBi adds a fixed negative bias proportional to the
distance between query and key positions directly to the attention scores before softmax:
$\text{score}_{ij} = q_i^\top k_j / \sqrt{d_k} - m \cdot |i - j|$, where $m$ is a
head-dependent slope.  This naturally decays attention with distance and extrapolates
to longer sequences.

**C.** ALiBi trains a separate positional embedding for each attention head.

**D.** ALiBi encodes position using binary representations of the position index.

---

### Q16 — Parameter count of an MoE model

A dense model has $N$ total parameters.  An MoE model has the same architecture but replaces
each FFN with $E = 8$ expert FFNs and a top-2 router.  Approximately how do the total parameter
counts compare?

**A.** MoE has $E$ times more FFN parameters but the same attention parameters, so total
parameters scale by roughly $E$ for a model where FFNs dominate.

**B.** MoE uses the same total parameters as the dense model because routing selects only 2
experts.

**C.** MoE has fewer parameters than the dense model because redundant experts share weights.

**D.** MoE doubles the attention parameters but keeps the FFN parameters the same.

---

### Q17 — Pre-norm in decoder-only LLMs

LLaMA uses RMSNorm applied BEFORE each sublayer (pre-norm).  Which of the following is an
accurate statement about pre-norm at large depth?

**A.** Pre-norm prevents the residual stream from growing large because the norm is applied
after the residual addition.

**B.** In pre-norm, the residual stream accumulates unnormalised updates, allowing its magnitude
to grow as the number of layers increases.  This can be beneficial (each layer adds incremental
information) but requires careful initialisation (e.g., scaling residual branch outputs by
$1/\sqrt{2L}$ or using $\mu P$ initialisation) to prevent divergence at initialisation.

**C.** Pre-norm and post-norm produce mathematically identical forward passes.

**D.** Pre-norm introduces a data-dependent normalisation that acts as an implicit attention
mask.

---

### Q18 — FlashAttention-2 improvements

FlashAttention-2 (Dao, 2023) improved over FlashAttention-1.  Which change is among its key
contributions?

**A.** It reduces the arithmetic complexity from $O(T^2 d)$ to $O(T d \log T)$.

**B.** It restructures the loop order so the outer loop iterates over query blocks (not key/value
blocks), improving parallelism and reducing synchronisation overhead on modern GPUs, leading to
roughly 2x additional wall-clock speedup over FlashAttention-1.

**C.** It replaces the tiling strategy with a purely sparse attention pattern.

**D.** It requires FP16 input; FP32 inputs are converted automatically, increasing memory usage.

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
| 7  | B      |
| 8  | B      |
| 9  | B      |
| 10 | B      |
| 11 | B      |
| 12 | A      |
| 13 | B      |
| 14 | B      |
| 15 | B      |
| 16 | A      |
| 17 | B      |
| 18 | B      |

---

## Detailed Explanations

---

### Q1 — Decoder-only architecture

**Correct: B.**

Decoder-only models (GPT family, LLaMA, Falcon, Mistral, etc.) stack causal self-attention
blocks.  There is no separate encoder and no cross-attention.  The causal mask enforces
left-to-right generation.  This design is maximally simple for generation tasks and scales
predictably.

- **A is wrong.** Encoder blocks use bidirectional attention; decoder-only models use causal
  (masked) attention.
- **C is wrong.** Some efficient attention variants use convolutions, but the dominant decoder-only
  LLM family uses standard scaled dot-product attention.
- **D is wrong.** Virtually all modern LLMs include positional information, most commonly via
  RoPE or learned embeddings.

---

### Q2 — RoPE mechanism

**Correct: B.**

RoPE's key property follows from the rotation formula: $R_m^\top R_n = R_{n-m}$, so
$(R_m q)^\top (R_n k) = q^\top R_{n-m} k$.  The dot product is a function of $q$, $k$, and
the RELATIVE offset $n - m$ only.  This combines the benefits of relative position encoding
with the efficiency of absolute-position approaches (no modification to the attention score
formula structure).

- **A is wrong.** That describes additive positional encodings; RoPE multiplies (rotates) the
  vectors.
- **C is wrong.** RoPE does not affect the scaling factor.
- **D is wrong.** RoPE is computed in the attention projection step, not in a separate memory
  bank.

---

### Q3 — RoPE frequency spectrum

**Correct: B.**

The frequencies $\theta_i = 10000^{-2i/d}$ decrease geometrically from $\theta_0 = 1$ (fast
rotation, high positional sensitivity) to $\theta_{d/2 - 1} = 10000^{-1}$ (slow rotation, low
positional sensitivity).  This multi-scale structure mirrors sinusoidal positional encoding and
lets the model distinguish both nearby and distant tokens.

- **A is wrong.** RoPE does not apply a hard cutoff on attended positions; the rotation simply
  becomes nearly indistinguishable for very large offsets at low-frequency dimensions.
- **C is wrong.** Each pair of dimensions uses a distinct angle $m \cdot \theta_i$; they do not
  rotate identically.
- **D is wrong.** While RoPE and sinusoidal PE share a similar frequency design, the application
  is fundamentally different (rotation of query/key vectors vs. additive embedding), and they
  do not produce identical representations.

---

### Q4 — RoPE context extension

**Correct: B.**

Positional interpolation (PI) rescales each token position as $pos \leftarrow pos \cdot
(L_{\text{original}} / L_{\text{extended}})$.  For example, to extend from 4096 to 16384 tokens,
multiply all positions by $4096/16384 = 0.25$.  After a brief fine-tuning phase on long contexts,
models recover high performance because the rotation angles now fall in the range seen during
pretraining.

- **A is wrong.** The number of attention heads is not related to context length.
- **C is wrong.** There is no "learning rate during inference"; inference is a forward pass only.
- **D is wrong.** Replacing RoPE with sinusoidal encodings would require re-pretraining from
  scratch.

---

### Q5 — GQA mechanism

**Correct: B.**

MHA uses $H$ Q, K, V heads each.  MQA (Shazeer, 2019) uses $H$ query heads but a single K/V
head shared by all.  GQA interpolates: $G$ K/V groups, each shared by $H/G$ query heads.
GQA is used in LLaMA-3, Mistral, Gemma, and many other models as it provides a favourable
accuracy-efficiency trade-off.

- **A is wrong.** GQA does not add extra query heads; it reduces K/V heads.
- **C is wrong.** Global attention (one attention operation for the full sequence) is a different
  technique; GQA maintains per-head computation for queries.
- **D is wrong.** GQA retains the value projection; it only reduces the number of distinct K/V
  projections.

---

### Q6 — GQA memory savings

**Correct: B.**

KV cache size is proportional to the number of K/V heads.  With $H = 32$ query heads and
$G = 8$ K/V groups, the K/V heads are reduced by a factor of $H / G = 32 / 8 = 4$.
The KV cache is therefore $4 \times$ smaller than standard MHA.

- **A is wrong.** A 2x reduction would correspond to $G = 16$.
- **C is wrong.** An 8x reduction would correspond to $G = 4$.
- **D is wrong.** A 32x reduction would correspond to $G = 1$, i.e., MQA (Multi-Query Attention).

---

### Q7 — MoE sparsity

**Correct: B.**

With $k = 2$ and $E = 8$, exactly 2 of the 8 expert FFNs process each token.  The fraction of
FFN parameters activated is $2/8 = 1/4$.  This is the key efficiency advantage: the model has
$E \times$ more FFN parameters than a dense equivalent but uses only $k/E$ of them per token.

- **A is wrong.** $1/8$ would correspond to $k = 1$.
- **C is wrong.** Only the top-$k$ experts are evaluated; the remaining $E - k$ are never
  activated for that token.
- **D is wrong.** Sparsity is applied to the FFN layer; attention is typically dense in standard
  MoE architectures.

---

### Q8 — MoE router collapse

**Correct: B.**

Without regularisation, gradient descent causes the router to specialise: a few experts receive
most tokens, become better at those tokens, receive more gradient, and improve further --
a positive feedback loop.  The standard fix is an auxiliary load-balancing loss that penalises
deviations from uniform expert load (e.g., $\mathcal{L}_{\text{aux}} = \alpha \sum_e f_e \cdot
p_e$ where $f_e$ is the fraction of tokens routed to expert $e$ and $p_e$ is the mean routing
probability).

- **A is wrong.** Router weights do not typically collapse to zero; the issue is extreme
  concentration on a few experts.
- **C is wrong.** Experts are not guaranteed to receive identical gradients; in fact, the problem
  is that some receive more gradient than others.
- **D is wrong.** Gradients do flow through the hard top-$k$ selection in practice via
  straight-through estimation or by treating the routing weights as the differentiable component.

---

### Q9 — FlashAttention problem

**Correct: B.**

FlashAttention's contribution is IO-awareness, not FLOP reduction.  Standard attention computes
$QK^\top$ and writes the full $T \times T$ matrix to HBM before computing softmax and then
$\text{softmax}(QK^\top)V$, requiring $O(T^2)$ HBM reads/writes.  FlashAttention tiles the
computation into SRAM-sized blocks (typically 64--128 rows at a time), fusing the entire
attention operation into a single kernel pass that reads from HBM only $O(T)$ times in total.
The FLOPs are identical to standard attention.

- **A is wrong.** FlashAttention does NOT reduce FLOPs; it may even increase them slightly
  due to recomputation during the backward pass.
- **C is wrong.** FlashAttention does not affect parameter counts.
- **D is wrong.** FlashAttention enables processing sequences longer than SRAM precisely by
  tiling; it does not store the full attention matrix in SRAM.

---

### Q10 — FlashAttention online softmax

**Correct: B.**

The numerically stable online softmax tracks the running maximum $m_i$ and normalisation
constant $\ell_i$ per row.  When a new block of keys arrives with max $m_{\text{new}}$, the
accumulated result is rescaled by $e^{m_{\text{prev}} - m_{\text{new}}}$ and the new block's
contribution is added.  This is provably exact; no approximation is involved.

- **A is wrong.** FlashAttention is exact, not approximate; its outputs match standard attention
  to floating-point precision.
- **C is wrong.** Linear attention with random features is a separate line of work (Performer,
  etc.) that does approximate the softmax.
- **D is wrong.** Dropping rows of attention scores would produce an approximate, not exact,
  result.

---

### Q11 — RMSNorm

**Correct: B.**

Standard LayerNorm: $\hat{x}_i = (x_i - \mu) / \sqrt{\sigma^2 + \epsilon}$, then scales and
shifts.  RMSNorm: $\hat{x}_i = x_i / \text{RMS}(x)$ where $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}$.
The mean-centring step is removed.  Zhang & Sennrich showed this achieves similar downstream
performance with ~7--15% faster normalisation in practice.

- **A is wrong.** Both LayerNorm and RMSNorm use per-sample (not per-batch) statistics.
- **C is wrong.** Both normalise over the feature (hidden) dimension.
- **D is wrong.** RMSNorm still has a learnable scale parameter $\gamma$; the difference is the
  absence of a learnable bias and the mean-centring step.

---

### Q12 — SwiGLU

**Correct: A.**

SwiGLU (Shazeer, 2020) defines: $\text{SwiGLU}(x) = (x W_1 \odot \text{Swish}_\beta(x W_2)) W_3$
where $\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$ (with $\beta = 1$ in practice).  The
gating mechanism allows the network to learn which features to pass through, improving
expressiveness.  To maintain the same parameter count as a standard FFN, the hidden dimension
is usually reduced by a factor of $2/3$.

- **B is wrong.** SwiGLU is not a polynomial approximation; it is a gated linear unit.
- **C is wrong.** SwiGLU does not involve a softmax operation.
- **D is wrong.** "Switched Weight" is not what SwiGLU stands for; the name comes from Swish + GLU.

---

### Q13 — Sliding window attention

**Correct: B.**

With window size $w$, each token attends to $w$ neighbours, so total attention computation is
$O(T \cdot w)$ rather than $O(T^2)$.  For $w \ll T$ this is a major saving.  Mistral 7B uses
a sliding window of $w = 4096$ on a context of 32K tokens, reducing quadratic cost 64x.
Global tokens (like the CLS token in Longformer) can attend to the full sequence, partially
recovering long-range dependencies.

- **A is wrong.** Long-range dependencies can still be captured through multiple layers; tokens
  far apart interact transitively through intermediate tokens (similar to convolutional receptive
  fields).
- **C is wrong.** $O(T^2/w^2)$ is not the correct formula; the complexity is $O(T \cdot w)$,
  not $O(T^2/w^2)$.
- **D is wrong.** The window is fixed (not recalculated), so the computation is genuinely $O(T w)$.

---

### Q14 — KV sharing across layers

**Correct: B.**

In architectures like RWKV cross-layer attention or some recent "MLA" designs (e.g., DeepSeek-V2),
sharing K/V projections across groups of layers directly reduces (1) the number of learned
projection matrices and (2) the KV cache footprint at inference.  The trade-off is that layers
sharing K/V cannot independently learn to extract different relational structures from the same
input.

- **A is wrong.** KV sharing is orthogonal to the residual connection design.
- **C is wrong.** The backward pass must still compute gradients for all layers; sharing parameters
  changes the gradient accumulation but does not reduce the number of backward passes.
- **D is wrong.** Multiple heads in standard MHA already attend to the same key positions; sharing
  K/V projections is a parameter reduction technique, not an attention-pattern constraint.

---

### Q15 — ALiBi

**Correct: B.**

ALiBi adds no positional information to the token embeddings.  Instead, the attention logit
for heads $h$ attending from position $i$ to $j$ is modified to:
$q_i^\top k_j / \sqrt{d_k} - m_h \cdot |i - j|$, where $m_h$ is a fixed geometric sequence
of slopes.  This biases attention toward nearby tokens without modifying the embedding space.
Because the bias is purely a function of distance, it naturally generalises to positions not
seen during training (unlike learned positional embeddings).

- **A is wrong.** ALiBi explicitly avoids modifying token embeddings; that is its key design choice.
- **C is wrong.** ALiBi's slopes $m_h$ are fixed (not learned) and are a deterministic geometric
  sequence.
- **D is wrong.** Binary position encoding is a different (less common) method.

---

### Q16 — MoE parameter count

**Correct: A.**

An MoE layer replaces one FFN with $E$ identical-sized FFNs.  The attention parameters are
unchanged.  In most large Transformers, the FFN accounts for roughly $\frac{2}{3}$ of total
parameters (since the FFN expansion ratio is typically 4x).  The total MoE parameter count is
therefore roughly: $N_{\text{MoE}} \approx N_{\text{attn}} + E \cdot N_{\text{FFN}}$.  For
$E = 8$, the MoE model has many more total parameters than the dense model but uses only $k/E$
of them per token during inference.

- **B is wrong.** Routing selects $k$ experts for COMPUTATION but does not reduce the total
  number of stored parameters; all $E$ expert weight matrices still reside in memory.
- **C is wrong.** Experts do not share weights in a standard MoE; each has independent parameters.
- **D is wrong.** MoE affects the FFN, not the attention mechanism.

---

### Q17 — Pre-norm residual stream growth

**Correct: B.**

With pre-norm, the update rule is $x_{l+1} = x_l + F_l(\text{Norm}(x_l))$.  The residual $x_l$
is NOT normalised, so its norm can grow linearly with depth.  This is distinct from post-norm
($x_{l+1} = \text{Norm}(x_l + F_l(x_l))$) where the norm is bounded after each layer.
The $\mu P$ (maximal update parametrisation) framework and "scaled residuals" initialisation
(multiplying $F_l$ output by $1/\sqrt{2L}$) are standard mitigations.

- **A is wrong.** Pre-norm normalises BEFORE the sublayer, not after the residual addition; the
  residual stream itself is not normalised.
- **C is wrong.** Pre-norm and post-norm differ in where normalisation is applied; they are
  mathematically distinct forward passes.
- **D is wrong.** RMSNorm/LayerNorm normalise by learned or computed statistics, but they are
  not equivalent to attention masking.

---

### Q18 — FlashAttention-2 improvements

**Correct: B.**

FlashAttention-1 loops over key/value blocks in the outer loop and query blocks in the inner
loop.  FlashAttention-2 reverses this: iterating over query blocks in the outer loop allows the
inner loop's output accumulation to be parallelised across independent thread blocks, reducing
synchronisation.  It also reduces the number of non-matrix-multiplication FLOPs.  The result is
~2x wall-clock speedup over FlashAttention-1 on A100 GPUs.

- **A is wrong.** Neither version of FlashAttention reduces the $O(T^2 d)$ arithmetic complexity;
  that is not the bottleneck on modern hardware.
- **C is wrong.** FlashAttention-2 retains the tiling/blocking strategy of FlashAttention-1; it
  does not introduce sparsity.
- **D is wrong.** FlashAttention-2 supports FP16, BF16, and FP32 inputs natively; there is no
  automatic upcast.
