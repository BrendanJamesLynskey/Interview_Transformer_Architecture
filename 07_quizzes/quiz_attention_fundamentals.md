# Quiz: Attention Mechanisms Fundamentals

**18 multiple-choice questions** covering attention scaling, self-attention vs cross-attention,
causal masking, positional encodings, multi-head attention parameter counts, residual connections,
and layer normalisation placement.

Difficulty: Fundamentals through Intermediate.

---

## Questions

---

### Q1 — Scaling the dot product

In scaled dot-product attention the raw attention scores are divided by $\sqrt{d_k}$
before the softmax.  What problem does this scaling solve?

**A.** It prevents the attention weights from summing to more than one.

**B.** It prevents dot products from growing large in magnitude, which would push softmax into
regions where gradients are extremely small.

**C.** It normalises the queries and keys to unit vectors, equivalent to cosine similarity.

**D.** It ensures the output of the attention layer has unit variance.

---

### Q2 — Softmax saturation

Suppose $d_k = 64$ and the dot product between a query and a key has magnitude roughly
$\sqrt{64} = 8$ before scaling.  Why would omitting the $\frac{1}{\sqrt{d_k}}$ factor
cause training instability?

**A.** The dot product would be negative, making softmax undefined.

**B.** The dot product magnitude grows as $\sqrt{d_k}$; without scaling, large $d_k$ produces
scores with magnitude $\sim d_k$, driving softmax to near-zero or near-one values and
producing gradients close to zero everywhere.

**C.** Without scaling, the attention weights would not sum to one.

**D.** Without scaling, the self-attention operation becomes non-linear and cannot be
differentiated.

---

### Q3 — Self-attention vs cross-attention

Which statement correctly distinguishes self-attention from cross-attention?

**A.** Self-attention uses three separate weight matrices; cross-attention uses only one shared
weight matrix for Q, K, and V.

**B.** In self-attention, Q, K, and V all come from the same sequence; in cross-attention, Q
comes from one sequence (e.g., the decoder) while K and V come from a different sequence
(e.g., the encoder output).

**C.** Self-attention allows each token to attend only to itself; cross-attention allows
full bidirectional attention.

**D.** Cross-attention is only possible in encoder-only architectures.

---

### Q4 — Causal masking

In a decoder-only language model (such as GPT), a causal attention mask is applied before
the softmax.  What values does this mask add to positions the current query must NOT attend to?

**A.** $0$, so those positions contribute nothing to the weighted sum.

**B.** $-1$, to subtract their influence.

**C.** $-\infty$ (implemented as a very large negative number), so those scores become
effectively zero after softmax.

**D.** $+\infty$, so those positions receive all the attention weight.

---

### Q5 — Multi-head attention parameter count

A multi-head attention layer has model dimension $d_{\text{model}} = 512$ and $h = 8$ heads,
so $d_k = d_v = 64$.  Ignoring biases, how many learnable parameters does it have?

**A.** $3 \times 512 \times 64 = 98{,}304$

**B.** $4 \times 512^2 = 1{,}048{,}576$

**C.** $3 \times 8 \times 64^2 + 512^2 = 196{,}608 + 262{,}144 = 458{,}752$

**D.** $512 \times 512 = 262{,}144$

---

### Q6 — Role of the output projection $W^O$

After concatenating all heads in multi-head attention, the result is projected through $W^O \in
\mathbb{R}^{h d_v \times d_{\text{model}}}$.  Why is this projection necessary?

**A.** It makes the attention operation differentiable.

**B.** It applies a final non-linearity to mix the head outputs.

**C.** It linearly combines information from all heads back into the full model dimension,
allowing cross-head interaction that is impossible during the per-head projections.

**D.** It is not strictly necessary; it is included only to match parameter counts with the FFN.

---

### Q7 — Sinusoidal positional encoding

The original Transformer (Vaswani et al., 2017) uses sinusoidal positional encodings:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Which property makes sinusoidal encodings attractive?

**A.** They are computed by a small neural network, so the model can adjust them during training.

**B.** The dot product between encodings at positions $pos$ and $pos + k$ depends only on $k$,
not on the absolute positions, so the model can generalise to relative offsets.

**C.** Each dimension oscillates at the same frequency, making position easy to decode.

**D.** They guarantee that the positional encoding never aligns with any token embedding.

---

### Q8 — Learned vs sinusoidal positional encodings

What is the primary practical advantage of **learned** positional embeddings over sinusoidal
ones?

**A.** Learned embeddings allow the model to attend to positions beyond $d_{\text{model}}$.

**B.** Learned embeddings generalise perfectly to sequences longer than those seen during
training.

**C.** Learned embeddings can capture arbitrary position-dependent patterns in the training data
and often achieve slightly better performance on standard benchmarks.

**D.** Learned embeddings reduce the total parameter count compared to sinusoidal encodings.

---

### Q9 — Attention complexity

Standard (dense) self-attention on a sequence of length $T$ has what time and memory
complexity?

**A.** $O(T)$ time and $O(1)$ memory.

**B.** $O(T \log T)$ time and $O(T \log T)$ memory.

**C.** $O(T^2)$ time and $O(T^2)$ memory, because the full $T \times T$ attention matrix
must be computed and stored.

**D.** $O(T^2)$ time but only $O(T)$ memory, because the attention matrix is sparse.

---

### Q10 — Residual connections in Transformers

Residual connections (skip connections) are added around each sublayer.  Which of the following
is the PRIMARY benefit of residual connections in deep Transformers?

**A.** They double the effective number of parameters without additional computation.

**B.** They allow gradients to flow directly back through the network to early layers, mitigating
the vanishing gradient problem and enabling training of much deeper models.

**C.** They force the network to learn an identity function in every sublayer.

**D.** They replace the need for layer normalisation.

---

### Q11 — Pre-norm vs post-norm

Modern LLMs (GPT-2 onwards) use **pre-norm**: `x + sublayer(LayerNorm(x))`, whereas the
original Transformer used **post-norm**: `LayerNorm(x + sublayer(x))`.  Why did the field
largely switch to pre-norm?

**A.** Pre-norm uses fewer parameters than post-norm.

**B.** Post-norm requires a different learning rate for each layer.

**C.** Pre-norm ensures the residual path carries unnormalised gradients, making training
more stable and generally eliminating the need for learning rate warm-up with very deep models.

**D.** Post-norm produces better final perplexity, so pre-norm was abandoned.

---

### Q12 — Layer normalisation vs batch normalisation

Why is **layer normalisation** preferred over **batch normalisation** in Transformer language
models?

**A.** Layer normalisation is faster to compute than batch normalisation.

**B.** Batch normalisation requires sequences of equal length and normalises across the batch
dimension, making it unreliable at small batch sizes or with variable-length text; layer
normalisation normalises across the feature dimension independently per sample.

**C.** Layer normalisation uses a larger number of learnable parameters, increasing model
capacity.

**D.** Batch normalisation cannot be applied to discrete token inputs.

---

### Q13 — Role of the FFN sublayer

Each Transformer layer contains a position-wise feed-forward network (FFN) applied after the
attention sublayer.  What function does the FFN primarily serve?

**A.** It provides a second stage of token-to-token interaction that the attention sublayer
cannot perform.

**B.** It applies a non-linear per-token transformation that allows the model to store and
retrieve factual knowledge in its weights; attention aggregates context, FFN processes it.

**C.** It projects the hidden state back to the vocabulary at every layer.

**D.** It reduces the sequence length from $T$ to $T/2$ to save computation.

---

### Q14 — Cross-attention in encoder-decoder models

In a standard encoder-decoder Transformer (e.g., the original Transformer for machine
translation), what do the cross-attention Keys and Values come from?

**A.** The previous decoder layer's output, identical to masked self-attention.

**B.** The encoder's final hidden states, so the decoder can attend over all source tokens.

**C.** A learnable memory matrix that is the same for all inputs.

**D.** The target token embeddings only.

---

### Q15 — Causal mask shape

For a decoder processing a sequence of $T$ tokens in a single forward pass (training mode),
what is the shape and structure of the causal attention mask?

**A.** A $T \times T$ matrix where the upper triangle (excluding the diagonal) is $-\infty$ and
all other values are $0$.

**B.** A $T \times T$ matrix where the lower triangle (including the diagonal) is $-\infty$.

**C.** A $1 \times T$ vector applied uniformly to all query positions.

**D.** A $T \times T$ matrix of all zeros -- no masking is needed during training.

---

### Q16 — Attention head specialisation

Multi-head attention uses $h$ parallel attention functions.  Which empirical observation
motivates this design?

**A.** Parallel heads allow the model to process multiple sequences simultaneously.

**B.** Each head can learn to focus on a different type of relationship
(e.g., syntactic dependencies, coreference, positional proximity), allowing the model to
jointly attend to information from different representation subspaces.

**C.** Parallel heads reduce the $O(T^2)$ attention complexity to $O(T^2 / h)$.

**D.** Heads are required to prevent gradient explosion in deep networks.

---

### Q17 — Dot product vs additive attention

The original seq2seq attention (Bahdanau et al., 2015) uses additive attention:
$\text{score}(q, k) = v^\top \tanh(W_1 q + W_2 k)$.
The Transformer uses scaled dot-product attention: $\text{score}(q, k) = q^\top k / \sqrt{d_k}$.
What is the main practical advantage of the dot-product form?

**A.** Dot-product attention is theoretically more expressive than additive attention.

**B.** Dot-product attention can be computed as a batch matrix multiplication, making it
significantly faster and more memory-efficient on modern hardware.

**C.** Dot-product attention does not require the scaling factor $\sqrt{d_k}$.

**D.** Dot-product attention works with sequences of length greater than 512; additive attention
does not.

---

### Q18 — Attention weight interpretation

After applying softmax, the attention weights $\alpha_{ij}$ for a given query position $i$ sum
to one across all key positions $j$.  What is the output of the attention operation for
position $i$?

**A.** The key vector with the highest attention weight.

**B.** The query vector at position $i$, unchanged.

**C.** A convex combination (weighted average) of the Value vectors, where the weights are the
softmax attention scores.

**D.** The concatenation of all Value vectors, weighted by $\alpha_{ij}$.

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | B      |
| 2  | B      |
| 3  | B      |
| 4  | C      |
| 5  | B      |
| 6  | C      |
| 7  | B      |
| 8  | C      |
| 9  | C      |
| 10 | B      |
| 11 | C      |
| 12 | B      |
| 13 | B      |
| 14 | B      |
| 15 | A      |
| 16 | B      |
| 17 | B      |
| 18 | C      |

---

## Detailed Explanations

---

### Q1 — Scaling the dot product

**Correct: B.**

When queries and keys are $d_k$-dimensional random vectors with zero mean and unit variance, their
dot product has variance $d_k$.  For large $d_k$ (e.g., 64 or 512), the raw dot products can be
large in magnitude, pushing the softmax into regions where its gradient is near zero -- the
so-called "softmax saturation" problem.  Dividing by $\sqrt{d_k}$ scales the dot products back to
unit variance, keeping gradients healthy.

- **A is wrong.** Softmax always produces a probability distribution summing to one regardless of
  the input magnitudes.
- **C is wrong.** Dividing by $\sqrt{d_k}$ does NOT normalise the vectors to unit norm; that would
  require dividing each vector by its own $\ell_2$ norm.
- **D is wrong.** The scaling controls gradient flow; it does not guarantee unit variance in the
  attention output (which depends on the value vectors as well).

---

### Q2 — Softmax saturation

**Correct: B.**

If $d_k = 64$, the unscaled dot product magnitude is approximately $64$ (variance = $d_k$ implies
std = $\sqrt{d_k}$, but we square for the full product).  A score of 64 vs. 0 in a softmax
effectively sends the weight to 1 for the maximum and 0 for everything else.  The gradient of
softmax approaches zero there.  Scaling by $1/\sqrt{64} = 1/8$ brings the scores back to a
reasonable range.

- **A is wrong.** Softmax is defined for all real inputs, including negative ones.
- **C is wrong.** Softmax always produces a valid probability distribution; scaling doesn't change
  that property.
- **D is wrong.** Self-attention is differentiable regardless of scaling; the issue is gradient
  magnitude, not existence.

---

### Q3 — Self-attention vs cross-attention

**Correct: B.**

The defining distinction is the source of Q, K, V:
- **Self-attention:** all three come from the same sequence (the encoder input, decoder input, etc.).
- **Cross-attention:** Q comes from one sequence, K and V from another.  In seq2seq, the decoder
  generates Q and the encoder output provides K and V.

- **A is wrong.** Both self-attention and cross-attention use three separate projection matrices
  $W^Q$, $W^K$, $W^V$.
- **C is wrong.** The "self" in self-attention refers to the source of the input, not the pattern
  of attention; self-attention can still attend broadly across the sequence.
- **D is wrong.** Cross-attention is a defining component of encoder-decoder architectures, and
  it also appears in other settings (e.g., latent variable models).

---

### Q4 — Causal masking

**Correct: C.**

Adding $-\infty$ (in practice, a very large negative float like $-10^9$ or $-\infty$ in fp16) to
masked positions means those positions receive $e^{-\infty} = 0$ weight after softmax.  This
prevents future tokens from influencing the current prediction.

- **A is wrong.** Adding $0$ leaves the score unchanged; future tokens would still receive
  attention weight.
- **B is wrong.** Subtracting 1 only slightly reduces the score; future tokens would still
  receive significant attention weight.
- **D is wrong.** Adding $+\infty$ would force ALL attention to the masked (future) positions,
  which is the opposite of causal masking.

---

### Q5 — Multi-head attention parameter count

**Correct: B.**

The four projection matrices are $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$
and $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ (since $h \cdot d_v = d_{\text{model}}$).
Each has $512 \times 512 = 262{,}144$ parameters, giving $4 \times 262{,}144 = 1{,}048{,}576$.

- **A is wrong.** This counts only $W^Q$ for a single head, ignoring the other three matrices.
- **C is wrong.** This incorrectly computes per-head matrices separately instead of recognising
  that the full $W^Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ already
  encompasses all heads.
- **D is wrong.** $512^2$ is the count for one projection matrix only; there are four in total.

---

### Q6 — Role of $W^O$

**Correct: C.**

Each head operates in a $d_k$-dimensional subspace.  Concatenating the $h$ head outputs gives a
$h \cdot d_v = d_{\text{model}}$-dimensional vector, but the heads have operated independently with
no interaction.  $W^O$ performs a final linear mixing that allows information from different heads
to be combined into a coherent representation before the residual addition.

- **A is wrong.** The attention operation is differentiable regardless of whether $W^O$ is present.
- **B is wrong.** $W^O$ is a linear (not non-linear) transformation; non-linearity comes from the
  FFN.
- **D is wrong.** $W^O$ has a clear functional role; removing it would measurably hurt performance.

---

### Q7 — Sinusoidal positional encoding

**Correct: B.**

A key property of the sinusoidal encoding is that $PE_{pos+k}$ can be written as a linear
function of $PE_{pos}$ (using the angle addition formula for sin/cos).  This means the dot product
between two position encodings depends only on their offset $k$, giving the model an inductive bias
toward relative position.

- **A is wrong.** Sinusoidal encodings are fixed mathematical functions, not learnable.
- **C is wrong.** Each dimension uses a different frequency ($10000^{-2i/d_{\text{model}}}$),
  ranging from high frequency (small $i$) to low frequency (large $i$).
- **D is wrong.** The encodings are added to token embeddings, so alignment is perfectly possible
  and in fact desirable for position-sensitive tasks.

---

### Q8 — Learned vs sinusoidal positional encodings

**Correct: C.**

Learned positional embeddings are tuned end-to-end and can capture dataset-specific positional
statistics.  Empirically, on standard NLP benchmarks the difference is small, but learned embeddings
often win slightly.

- **A is wrong.** Neither encoding scheme has a built-in limit on the number of attended positions;
  both are indexed by an integer position.
- **B is wrong.** This is the OPPOSITE of the truth.  Learned embeddings only cover positions seen
  during training; sinusoidal encodings can be extrapolated (though not perfectly) to longer
  sequences because they are computed analytically.
- **D is wrong.** Learned embeddings ADD parameters ($\text{max\_seq\_len} \times d_{\text{model}}$
  extra); sinusoidal encodings have zero parameters.

---

### Q9 — Attention complexity

**Correct: C.**

Computing the $T \times T$ attention score matrix requires $O(T^2 \cdot d_k)$ operations, and
storing it requires $O(T^2)$ memory.  This quadratic scaling is the central motivation for
efficient attention methods (Longformer, BigBird, FlashAttention, etc.).

- **A is wrong.** $O(T)$ attention would require strong sparsity assumptions not present in
  standard dense attention.
- **B is wrong.** $O(T \log T)$ complexity is characteristic of certain approximate attention
  methods, not standard dense attention.
- **D is wrong.** Standard attention is not sparse; the full $T \times T$ matrix is materialised
  in memory (though FlashAttention avoids this by recomputing in tiles).

---

### Q10 — Residual connections

**Correct: B.**

The residual (skip) connection creates a direct gradient highway from the loss back to early
layers, bypassing the sublayer transformations.  This was the key insight behind ResNets (He et al.,
2016) and is equally critical for deep Transformers.  Without residuals, gradients in very deep
networks diminish exponentially (vanishing gradient problem).

- **A is wrong.** Residual connections do not add parameters; the sublayer still has its own weights.
- **C is wrong.** The residual does not force an identity; it adds the original input to the sublayer
  output, so the sublayer only needs to learn the residual (difference).
- **D is wrong.** Layer normalisation and residual connections serve different purposes and are
  both used together.

---

### Q11 — Pre-norm vs post-norm

**Correct: C.**

With post-norm, both the sublayer output and the residual pass through LayerNorm, which can
cause the residual stream to shrink at initialisation, making the effective depth small.
Pre-norm keeps the residual path free of normalisation, so gradients from any depth flow cleanly
through the skip connections.  This makes training more robust without requiring careful warm-up,
especially at large scales.

- **A is wrong.** Pre-norm and post-norm use the same number of LayerNorm parameters.
- **B is wrong.** Warm-up may still be used with pre-norm, but it is less critical.
- **D is wrong.** Post-norm sometimes achieves slightly better final performance if trained carefully,
  but the stability advantages of pre-norm have made it dominant in practice.

---

### Q12 — Layer normalisation vs batch normalisation

**Correct: B.**

Batch normalisation computes statistics over the batch and sequence dimensions.  In NLP this is
problematic: sequences vary in length, batch sizes are small (limited by GPU memory), and the model
must run autoregressively with a batch size of one at inference time.  Layer normalisation computes
statistics over the feature (model dimension) axis for each token independently, avoiding all these
issues.

- **A is wrong.** Layer normalisation is not necessarily faster; both have similar $O(d_{\text{model}})$
  cost per token.
- **C is wrong.** Both methods have the same number of learnable parameters (scale and bias per
  feature dimension).
- **D is wrong.** Batch normalisation can in principle be applied after embedding lookup; its
  failure mode in NLP is statistical, not representational.

---

### Q13 — Role of the FFN sublayer

**Correct: B.**

Attention aggregates contextual information across positions; the FFN applies a position-wise
non-linear transformation that is the same for every token.  Research (e.g., Geva et al., 2021
"Transformer Feed-Forward Layers Are Key-Value Memories") has shown that FFN layers store factual
associations: the keys correspond to input patterns and the values promote output tokens.

- **A is wrong.** The FFN operates independently on each token position; it does NOT model
  token-to-token interaction.
- **C is wrong.** Vocabulary projection is done only at the final layer (the LM head), not at
  each FFN.
- **D is wrong.** FFNs in the original Transformer maintain the sequence length $T$ throughout.

---

### Q14 — Cross-attention in encoder-decoder models

**Correct: B.**

In the original Transformer, the decoder's cross-attention layer projects the decoder state to Q,
then attends over the encoder's final hidden states used as K and V.  This allows every decoder
position to "look up" relevant source information regardless of position.

- **A is wrong.** That describes the decoder's separate masked self-attention layer.
- **C is wrong.** A learnable memory matrix does appear in some architectures (e.g., memory-augmented
  Transformers) but not in the standard encoder-decoder Transformer.
- **D is wrong.** Target embeddings are used as the decoder's input to its self-attention layer,
  not as K/V for cross-attention.

---

### Q15 — Causal mask shape

**Correct: A.**

The causal mask is a $T \times T$ matrix where position $(i, j)$ receives $-\infty$ if $j > i$
(i.e., key position $j$ is in the future relative to query position $i$).  The diagonal ($j = i$)
is NOT masked because each token must attend to itself.

- **B is wrong.** Masking the lower triangle would prevent attending to past tokens, which is the
  opposite of causal masking.
- **C is wrong.** A vector is insufficient because each query position has its own distinct set of
  allowed key positions.
- **D is wrong.** Without any masking, the model would attend to future tokens, causing information
  leakage during training.

---

### Q16 — Attention head specialisation

**Correct: B.**

Interpretability research (e.g., Clark et al., 2019; Voita et al., 2019) has identified heads
that track specific syntactic and semantic roles.  The low-rank per-head projections allow each
head to focus on a different subspace of the representation.

- **A is wrong.** Heads operate within a single forward pass on a single sequence; they do not
  process separate sequences.
- **C is wrong.** Each head still computes $O(T^2)$ attention scores; the total complexity is
  $O(h \cdot T^2 \cdot d_k)$, which for fixed $d_{\text{model}}$ equals $O(T^2 \cdot
  d_{\text{model}})$ -- no reduction.
- **D is wrong.** Gradient explosion is managed by residual connections, layer normalisation,
  and gradient clipping -- not by the number of heads.

---

### Q17 — Dot product vs additive attention

**Correct: B.**

The key computational advantage is that scaled dot-product attention reduces to matrix
multiplication: $QK^\top$.  Modern GPUs and TPUs have highly optimised BLAS kernels for matrix
multiplication.  Additive attention requires an elementwise $\tanh$ and cannot be expressed as
a pure matrix product, making it roughly 3--5x slower in practice.

- **A is wrong.** Theoretical expressiveness is similar between the two forms; the practical
  difference is efficiency, not representational power.
- **C is wrong.** The scaling by $\sqrt{d_k}$ is required precisely for dot-product attention;
  additive attention implicitly normalises via the learned $v$ and $W$ parameters.
- **D is wrong.** Both attention mechanisms are applicable to any sequence length; the limitation
  is computational cost ($O(T^2)$), not a hard constraint imposed by the scoring function.

---

### Q18 — Attention weight interpretation

**Correct: C.**

The output for position $i$ is:

$$\text{out}_i = \sum_j \alpha_{ij} V_j$$

where $\alpha_{ij} \geq 0$ and $\sum_j \alpha_{ij} = 1$.  This is a convex combination of Value
vectors, weighted by how relevant each key position is to the current query.

- **A is wrong.** The "hard" argmax attention is a non-differentiable limiting case; standard
  attention computes a soft weighted average over ALL Value vectors.
- **B is wrong.** The query is used to compute attention scores but the output is formed from
  Value vectors, not the query itself.
- **D is wrong.** Concatenation would produce a $T \cdot d_v$-dimensional output, whereas the
  attention output matches $d_v$ for each position -- a weighted SUM, not a concatenation.
