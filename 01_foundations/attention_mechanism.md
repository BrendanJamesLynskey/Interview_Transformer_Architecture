# Attention Mechanism

## Overview

Attention is the central innovation of the Transformer architecture. It allows a model to dynamically weight the relevance of different parts of the input when producing each output element. Understanding attention deeply — its intuition, its variants, and why it displaced recurrence — is essential for any serious ML role.

---

## Tier 1: Fundamentals

### Q1. What problem does attention solve, and what was the dominant approach before it?

**Answer.**

Before attention, sequence-to-sequence tasks (translation, summarisation) used RNN-based encoder-decoder architectures. The encoder compressed the entire source sequence into a single fixed-size hidden state vector, which the decoder then used to generate the output. This creates an **information bottleneck**: regardless of source sequence length, the decoder receives exactly the same amount of information. Empirically, BLEU scores degraded significantly for sentences longer than ~20 tokens.

Attention (Bahdanau et al., 2015) broke the bottleneck by allowing the decoder to look back at **all** encoder hidden states at each decoding step, forming a weighted combination. The weights are learned dynamically based on the relevance of each encoder position to the current decoder state.

The key insight is that when translating "the bank of the river", the word "bank" should attend strongly to "river" to resolve ambiguity — not rely on a single compressed vector to somehow encode this.

**Common mistake:** Confusing the attention mechanism with the Transformer architecture. Attention was introduced for RNNs and later became the backbone of the Transformer by eliminating recurrence entirely.

---

### Q2. Explain the query-key-value (QKV) analogy. What does each component represent?

**Answer.**

The QKV abstraction is best understood through a retrieval analogy.

Consider a **soft dictionary lookup**. In a hard dictionary:
- You have a **query** (what you are looking for)
- The dictionary has **keys** (the index entries)
- Each key maps to a **value** (the stored content)

You look up the key that exactly matches your query and return its value.

Attention makes this **soft**: instead of finding an exact match, you compute a similarity score between the query and every key, convert those scores to a probability distribution (via softmax), and return a **weighted sum of all values**.

In a self-attention layer of a Transformer:
- **Query ($Q$)**: the token asking "what information do I need?"
- **Key ($K$)**: each token advertising "what information do I contain?"
- **Value ($V$)**: the actual content each token contributes if selected

For token $i$ attending over a sequence:
1. Compute relevance of token $i$ to every token $j$: dot product $q_i \cdot k_j$
2. Normalise into weights: softmax over all $j$
3. Return weighted sum of values: $\sum_j \alpha_{ij} v_j$

**Why separate keys and values?** A key can be optimised purely for matching (like an index), while the value carries the semantic content to be aggregated. This decoupling gives the model more representational flexibility than using a single vector for both purposes.

---

### Q3. Why did attention replace RNNs and LSTMs as the dominant architecture for sequence modelling?

**Answer.**

Three fundamental limitations of RNNs drove this transition:

**1. Sequential computation prevents parallelism.**
RNNs process tokens one at a time: hidden state $h_t$ depends on $h_{t-1}$. This means an $n$-token sequence requires $n$ sequential steps. Modern hardware (GPUs, TPUs) is designed for massively parallel matrix operations. Transformers compute all token interactions simultaneously in a single matrix multiply, achieving far higher hardware utilisation.

**2. The vanishing/exploding gradient problem.**
Gradients must propagate through $n$ timesteps during backpropagation through time (BPTT). For long sequences, gradients either vanish (information from early tokens is lost) or explode. LSTMs mitigated but did not eliminate this. Attention creates **direct connections** between any two positions, so gradient path length is $O(1)$ regardless of sequence length.

**3. Fixed-length bottleneck (for seq2seq).**
As discussed in Q1, the single context vector constrains what the decoder can access.

**Additional advantage:** Attention is interpretable. Attention weights directly show which source tokens a given output token is attending to, providing some level of model introspection.

**Trade-off to acknowledge:** Self-attention has $O(n^2)$ memory and compute complexity in sequence length $n$, while RNNs are $O(n)$. For very long sequences (e.g., 100k tokens), this becomes a real constraint that motivates work like Longformer, FlashAttention, and linear attention variants.

---

### Q4. What is the difference between self-attention and cross-attention?

**Answer.**

The difference lies in **where the queries, keys, and values come from**.

**Self-attention**: Q, K, and V all come from the same sequence.
- Each token attends to every other token in the same sequence.
- Used in the Transformer encoder to build context-aware representations.
- Also used in the Transformer decoder (with a causal mask to prevent attending to future tokens).

**Cross-attention**: Q comes from one sequence; K and V come from a different sequence.
- The decoder's cross-attention layers let each decoder position query the encoder's output.
- Q = decoder hidden states, K = V = encoder output.
- This is the mechanism that conditions generation on the source sequence in translation, for example.

Formally, both compute:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The formula is identical; only the source of the matrices differs.

**In modern LLMs (decoder-only architectures like GPT):** There is no cross-attention at all. Only causal self-attention is used. Cross-attention appears in encoder-decoder models (T5, original Transformer, Whisper).

---

### Q5. What is meant by "attention as soft dictionary lookup"?

**Answer.**

A hard dictionary maps a query to exactly one key-value pair (or returns null if not found). Attention generalises this to a **differentiable, probabilistic** version:

Given:
- Query $q \in \mathbb{R}^{d_k}$
- Keys $K \in \mathbb{R}^{n \times d_k}$ (one key per entry)
- Values $V \in \mathbb{R}^{n \times d_v}$ (one value per entry)

The retrieval proceeds as:
1. Compute similarity scores: $s_i = q \cdot k_i$ for all $i$
2. Normalise to a distribution: $\alpha_i = \text{softmax}(s_i / \sqrt{d_k})$
3. Return weighted aggregate: $\text{output} = \sum_i \alpha_i v_i$

Properties that make this "soft":
- **All** values contribute to the output (with varying weights)
- The weighting is differentiable, so gradients flow through to Q, K, V
- With temperature scaling ($\sqrt{d_k}$ in the denominator), you can interpolate between uniform attention (very soft) and nearly hard one-hot attention (very sharp)

This framing makes it clear why the operation is powerful: the model can retrieve a **blend** of information from multiple positions rather than committing to one.

---

## Tier 2: Intermediate

### Q6. Walk through the mathematical derivation of how attention weights are computed for a single query attending over $n$ keys. What are the shapes at each step?

**Answer.**

Assume:
- Single query $q \in \mathbb{R}^{d_k}$
- Key matrix $K \in \mathbb{R}^{n \times d_k}$ (n tokens, each with a $d_k$-dimensional key)
- Value matrix $V \in \mathbb{R}^{n \times d_v}$

**Step 1: Compute raw scores.**

$$e_i = \frac{q \cdot k_i}{\sqrt{d_k}}, \quad e \in \mathbb{R}^n$$

As a matrix operation: $e = \frac{qK^T}{\sqrt{d_k}}$, shape $(1 \times n)$.

**Step 2: Apply softmax.**

$$\alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^{n} \exp(e_j)}, \quad \alpha \in \mathbb{R}^n, \quad \sum_i \alpha_i = 1$$

**Step 3: Weighted sum of values.**

$$\text{output} = \sum_{i=1}^{n} \alpha_i v_i = \alpha V, \quad \text{shape } (1 \times d_v)$$

**Batched form** (all queries at once), $Q \in \mathbb{R}^{n \times d_k}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Shapes: $QK^T \in \mathbb{R}^{n \times n}$, softmax output $\in \mathbb{R}^{n \times n}$, final output $\in \mathbb{R}^{n \times d_v}$.

**Why $\sqrt{d_k}$ scaling matters:** If $q$ and $k$ are i.i.d. with zero mean and unit variance, each component of their dot product has variance 1. The dot product $q \cdot k = \sum_{j=1}^{d_k} q_j k_j$ has variance $d_k$. Without scaling, for large $d_k$, the dot products grow large, pushing softmax into saturation regions where gradients are near zero. Dividing by $\sqrt{d_k}$ restores unit variance.

---

### Q7. Why does the causal (autoregressive) mask take the form it does? What happens computationally when masking is applied?

**Answer.**

In a decoder generating tokens left-to-right, position $i$ must not attend to position $j > i$ (future tokens). This is enforced by the **causal mask**, an upper-triangular matrix of $-\infty$ values.

Before softmax, the raw attention score matrix is:

$$E = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

The causal mask $M$ is:

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Applying the mask: $\tilde{E} = E + M$

After softmax, $\exp(-\infty) = 0$, so masked positions receive exactly zero attention weight. The softmax denominator only sums over valid (past and current) positions.

```
Position: 0  1  2  3
          [0  -∞ -∞ -∞]   <- token 0 attends only to itself
          [0   0  -∞ -∞]   <- token 1 attends to 0, 1
          [0   0   0  -∞]  <- token 2 attends to 0, 1, 2
          [0   0   0   0]   <- token 3 attends to all
```

**Key properties:**
- Causality is enforced **without any sequential computation** — the entire sequence is still processed in parallel during training
- At inference time for autoregressive generation, you generate one token at a time (or use KV caching), so the mask is effectively applied by construction
- The mask must be applied before softmax, not after, because softmax normalisation must be performed over the unmasked subset only

---

### Q8. Explain the difference between additive attention (Bahdanau) and dot-product attention (Luong/Transformer). When might you prefer one over the other?

**Answer.**

**Additive attention (Bahdanau, 2015):**

$$e_{ij} = v^T \tanh(W_q q_i + W_k k_j)$$

A learned feed-forward network computes compatibility. Parameters: $W_q \in \mathbb{R}^{d \times d_q}$, $W_k \in \mathbb{R}^{d \times d_k}$, $v \in \mathbb{R}^d$.

**Dot-product attention (Transformer):**

$$e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

No additional parameters (aside from the linear projections to Q, K, V).

**Comparison:**

| Property | Additive | Dot-product |
|---|---|---|
| Parameters | Yes ($W_q, W_k, v$) | No (in the score function itself) |
| Speed | Slower (sequential tanh) | Faster (matrix multiply, hardware-optimised) |
| Scaling | Less sensitive to $d_k$ | Needs $\sqrt{d_k}$ scaling |
| Performance | Comparable for small $d_k$ | Better for large $d_k$ with scaling |

Empirically, scaled dot-product attention matches or exceeds additive attention and is far more computationally efficient. This is why it was chosen for the Transformer. Modern implementations can leverage BLAS matrix routines and fused attention kernels (e.g., FlashAttention) which are highly optimised for dot products.

---

### Q9. What does it mean for attention to be "permutation equivariant" and what consequence does this have for the architecture?

**Answer.**

**Permutation equivariance** means: if you permute the input tokens, the output tokens are permuted in the same way. Formally, for a permutation $\pi$:

$$\text{Attention}(\pi Q, \pi K, \pi V) = \pi \cdot \text{Attention}(Q, K, V)$$

This follows from the fact that dot-product attention is symmetric in how it treats positions — the score between token $i$ and token $j$ depends only on their feature vectors, not on where in the sequence they appear.

**Consequence:** A pure attention mechanism is **positionally blind**. The model would produce the same output regardless of token order: "cat sat mat" and "mat cat sat" are indistinguishable.

This is why **positional encodings are mandatory**. By adding (or concatenating) position information to the input embeddings before attention, you break the permutation equivariance and allow the model to distinguish positions.

Contrast with RNNs, which are inherently position-aware because the hidden state is updated sequentially — position is encoded implicitly in the order of processing.

**Note:** This is actually a strength for some applications (e.g., set-based problems where order genuinely doesn't matter), but for language, order is semantically critical.

---

## Tier 3: Advanced

### Q10. Describe the relationship between attention and the Hopfield network / associative memory. What does this connection reveal?

**Answer.**

Ramsauer et al. (2020) showed that modern (dense) Hopfield networks and scaled dot-product attention are mathematically equivalent.

A modern Hopfield network with stored patterns $\xi_1, ..., \xi_n$ retrieves the pattern most similar to a query $q$ via the update rule:

$$q_{\text{new}} = \Xi \, \text{softmax}(\beta \Xi^T q)$$

where $\Xi = [\xi_1, ..., \xi_n]$ and $\beta$ is an inverse temperature.

Compare with attention: $\text{Attention}(Q, K, V) = V \cdot \text{softmax}(K^T Q / \sqrt{d_k})$.

These are structurally identical with $\beta = 1/\sqrt{d_k}$, $K \leftrightarrow \Xi$, $V \leftrightarrow \Xi$.

**What this reveals:**

1. **Storage capacity:** Modern Hopfield networks can store exponentially many patterns (in $d_k$). This explains why Transformers can store vast associative memories in their weights and context.

2. **Retrieval as pattern completion:** Each attention operation is fundamentally retrieving a stored pattern based on partial query information. This reframes the attention computation as a form of associative memory lookup.

3. **Multiple heads:** Multiple heads correspond to multiple retrieval operations — each head can specialise in retrieving different types of patterns (syntactic, semantic, positional, etc.).

4. **Temperature interpretation:** The scaling factor $1/\sqrt{d_k}$ controls retrieval sharpness. Larger $d_k$ → lower temperature → sharper (more focused) retrieval.

---

### Q11. How does attention complexity scale, and what approaches exist to reduce it for long sequences?

**Answer.**

**Standard self-attention complexity:**
- Time: $O(n^2 d)$ — $n^2$ pairwise dot products, each of dimension $d$
- Memory: $O(n^2)$ — must store the full attention matrix

For $n = 100{,}000$ tokens: $n^2 = 10^{10}$ operations — computationally infeasible at full precision.

**Approaches to sub-quadratic attention:**

**1. Sparse attention (Longformer, BigBird):**
Restrict each token to attend only to a local window plus a few global tokens. Complexity: $O(nw)$ where $w$ is window size. Loses full pairwise connectivity but works well for tasks with local structure.

**2. Linear attention (Katharopoulos et al., 2020):**
Approximate $\text{softmax}(QK^T)$ by decomposing the kernel: $\phi(Q)\phi(K)^T$ where $\phi$ is a feature map. Use the associativity of matrix multiplication:

$$(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V)$$

Computing $\phi(K)^T V$ first is $O(nd^2)$, then multiplying by $\phi(Q)$ is $O(nd^2)$. Total: $O(nd^2)$, linear in sequence length. Trade-off: approximation quality varies.

**3. Flash Attention (Dao et al., 2022):**
Not a mathematical approximation but an **IO-aware exact algorithm**. Tiles the computation to avoid materialising the full $n \times n$ attention matrix in HBM (GPU high-bandwidth memory). Uses online softmax to fuse operations into a single kernel pass. Achieves:
- Same mathematical result as standard attention
- $O(n^2)$ time but $O(n)$ memory (due to tiling)
- 2-4x wall-clock speedup from reduced memory bandwidth

**4. Sliding window / local attention (used in Mistral, Phi):**
Use full attention within a window of size $w$ and no attention outside. This is $O(nw)$ and works well because most relevant context is local.

**5. Multi-Query Attention / Grouped-Query Attention:**
Reduce KV cache memory (critical at inference) by sharing K and V across multiple heads. Not a complexity reduction in FLOPs but significantly reduces memory bandwidth during autoregressive generation.

---

### Q12. An interviewer asks: "In a trained Transformer, what are attention heads actually learning?" Summarise the empirical evidence and its limitations.

**Answer.**

**Empirical findings (what heads appear to do):**

Vig & Belinkov (2019), Clark et al. (2019), and others analysed attention patterns in BERT and GPT-2 and identified recurring patterns:

- **Syntactic heads:** Some heads align with dependency parse structure — e.g., a head that consistently attends from a verb to its subject.
- **Positional heads:** Some heads predominantly attend to adjacent tokens (position $i$ attends to $i-1$ or $i+1$), effectively acting as a local smoothing filter.
- **Rare-word heads:** Some heads attend from rare tokens to more common tokens, possibly for contextual disambiguation.
- **[CLS] token heads:** In BERT, some heads concentrate attention on the [CLS] token, aggregating global sequence information.
- **Delimiter heads:** Many heads attend to [SEP] tokens, which may serve as "no-op" attention (dumping attention weight somewhere harmless when no useful target exists).

**Evidence from ablation:**

Michel et al. (2019) showed that at test time, the majority of attention heads in BERT can be pruned with minimal performance loss — suggesting significant redundancy and that not all heads learn distinct patterns.

**Important limitations and caveats:**

1. **Attention $\neq$ explanation.** Jain & Wallace (2019) and Wiegreffe & Pinter (2019) debated whether attention weights constitute a faithful explanation. High attention weight on a token does not necessarily mean that token causally influenced the output — gradient-based attribution tells a different story.

2. **Representations, not routing.** After the attention-weighted sum, the model applies a dense projection ($W^O$) that mixes all heads. Individual head interpretations may be misleading.

3. **Layer dependence.** Early layers tend to capture local/syntactic patterns; later layers capture more semantic and task-specific patterns. There is no single universal behaviour.

4. **Training dynamics.** The same architectural slot can learn different patterns depending on initialisation, data, and task. Interpretations generalise across instances of the same model family but not universally.

**Bottom line for interviews:** You can cite specific findings but should acknowledge that attention interpretability is an active research area with fundamental open questions about whether visualising attention weights tells us what we think it does.
