# Decoder-Only LLMs

## Overview

Decoder-only transformers now dominate large language model development. Understanding why this architectural choice won out, and how successive models (GPT, LLaMA, Mistral, Gemma) refined it, is essential for any serious ML engineering or research interview.

---

## Fundamentals

### Q1. What are the three main transformer variants and what tasks are each suited for?

**Answer.**

The original Transformer paper introduced an encoder-decoder architecture for sequence-to-sequence tasks. Three families emerged:

| Variant | Attention Mask | Representative Models | Primary Use |
|---|---|---|---|
| Encoder-only | Bidirectional (full) | BERT, RoBERTa, DeBERTa | Classification, NER, embeddings |
| Encoder-decoder | Encoder: full; Decoder: causal | T5, BART, mT5 | Translation, summarisation, structured generation |
| Decoder-only | Causal (lower-triangular) | GPT series, LLaMA, Mistral | Open-ended generation, in-context learning |

The causal mask ensures that position $i$ can only attend to positions $j \leq i$:

$$M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$$

---

### Q2. What is the core GPT architectural recipe?

**Answer.**

GPT (Generative Pre-trained Transformer) introduced a clean decoder-only recipe:

1. **Token embedding** + **learned positional encoding**: $\mathbf{x} = \text{Embed}(t) + \text{PosEmbed}(pos)$
2. $N$ identical transformer decoder blocks, each containing:
   - Pre-norm (in later variants): LayerNorm before sub-layers
   - Causal multi-head self-attention
   - Residual connection
   - LayerNorm
   - Position-wise FFN with GELU activation: $\text{FFN}(\mathbf{x}) = W_2 \cdot \text{GELU}(W_1 \mathbf{x} + b_1) + b_2$
   - Residual connection
3. Final LayerNorm
4. Linear projection to vocabulary size (weight-tied with embedding matrix in GPT-2 onwards)

The training objective is standard language modelling — maximise $\sum_t \log P(x_t \mid x_{<t})$.

---

### Q3. Why does decoder-only dominate over encoder-decoder for large-scale language models?

**Answer.**

Several converging reasons explain the dominance:

**Simplicity and generality.** A decoder-only model handles any task as text-in, text-out. There is no architectural distinction between "input" and "output" tokens — both are modelled autoregressively. This unification means the same model handles translation, summarisation, code generation, and classification (via prompting) without task-specific heads.

**Scaling efficiency.** For a fixed parameter budget, a single decoder stack uses all parameters for both "understanding" the prompt and generating the response. An encoder-decoder splits the budget, with the encoder processing input it will never directly generate and the decoder being smaller.

**KV cache compatibility.** During inference, autoregressive decoding naturally caches key-value pairs for all previous tokens. Encoder-decoder models require a cross-attention KV cache for encoder outputs in addition to self-attention caches, increasing memory pressure.

**Emergent few-shot ability.** Brown et al. (GPT-3, 2020) showed that purely autoregressive pretraining at scale produces strong in-context learning without any architectural specialisation for it. The model learns to condition on examples in its context window.

**Training stability at scale.** A single objective (next-token prediction) on a single architecture is easier to scale reliably than joint objectives or architectures with separate components.

The primary advantage of encoder-decoder — bidirectional context for the input — turns out to be largely recoverable via prompting or by including the full input in the decoder's context window.

---

### Q4. What does "causal masking" mean and why is it necessary for language model training?

**Answer.**

Causal masking prevents each token from attending to future tokens during training. Without it, the model could trivially "cheat" by copying the answer from future positions when predicting the next token, learning nothing useful.

Mechanically, the attention logit matrix $A \in \mathbb{R}^{T \times T}$ is modified before softmax:

$$\tilde{A}_{ij} = \begin{cases} Q_i K_j^T / \sqrt{d_k} & j \leq i \\ -\infty & j > i \end{cases}$$

After softmax, any $-\infty$ entry becomes exactly $0$, so those positions contribute nothing to the output.

During training, causal masking also allows computing all $T$ next-token predictions in a single forward pass (since each position sees only its valid prefix), giving efficient parallelism. This is a crucial practical advantage over approaches that would require $T$ separate forward passes.

---

## Intermediate

### Q5. Describe the LLaMA architecture innovations over GPT-2/GPT-3. Why was each change made?

**Answer.**

LLaMA (Touvron et al., 2023) is a clean, highly optimised decoder-only model. The key departures from the GPT-2/3 recipe are:

**1. RMSNorm instead of LayerNorm.**

LayerNorm computes:
$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \cdot \gamma + \beta \quad \text{where } \mu = \frac{1}{d}\sum x_i,\; \sigma = \sqrt{\frac{1}{d}\sum(x_i - \mu)^2}$$

RMSNorm drops the mean-centring entirely:
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma \quad \text{where } \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum x_i^2}$$

This is approximately $7\text{--}15\%$ faster because it avoids computing the mean. The hypothesis (Zhang & Sennrich, 2019) is that re-centring is not necessary for training stability; only re-scaling matters. Empirically this holds at large scale.

**2. SwiGLU activation in the FFN.**

GPT uses GELU. LLaMA uses SwiGLU:
$$\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \otimes (xV + c)$$
$$\text{Swish}(\mathbf{z}) = \mathbf{z} \cdot \sigma(\mathbf{z})$$

The gated linear unit structure uses element-wise multiplication of two parallel projections — one passed through Swish, one linear. This was shown by Shazeer (2020) to improve perplexity. To keep parameter count constant vs a standard FFN, the hidden dimension is scaled by $\frac{2}{3}$ (since there are now three matrices: $W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$).

**3. Rotary Position Embeddings (RoPE).**

GPT-2 uses learned absolute position embeddings. LLaMA uses RoPE, which encodes position by rotating the query and key vectors in the attention computation. Key advantages: relative position bias is automatically captured, and the model extrapolates better to contexts longer than seen during training (see `rope_and_alibi.md` for full derivation).

**4. Pre-norm placement.**

Both GPT-2 and LLaMA use pre-norm (normalise before the sub-layer), but this is worth noting as the original Transformer used post-norm. Pre-norm improves gradient flow for very deep networks (see `rmsnorm_and_pre_norm.md`).

**5. No biases.**

LLaMA removes all linear layer biases, reducing parameter count slightly and simplifying the model without performance loss.

---

### Q6. What are the key innovations in Mistral 7B?

**Answer.**

Mistral 7B (Jiang et al., 2023) achieves performance exceeding LLaMA-2 13B at 7B parameters through two architectural innovations:

**Sliding Window Attention (SWA).**

Standard attention is $O(n^2)$ in sequence length. SWA restricts each token to attend only to the $W$ most recent tokens:

$$\text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K_{[i-W:i]}^T}{\sqrt{d_k}}\right) V_{[i-W:i]}$$

With $W = 4096$ in Mistral, any single layer can only "see" 4096 tokens. However, information propagates across layers: after $k$ layers, a token effectively has a receptive field of $k \cdot W$ tokens. For 32 layers with $W = 4096$, the effective context is 131,072 tokens.

SWA reduces memory and compute per attention operation and is implemented efficiently with a rolling KV cache buffer of size $W$.

**Grouped Query Attention (GQA).**

Mistral uses GQA with 8 KV heads and 32 query heads (ratio 4:1). This dramatically reduces KV cache memory during inference — 4x reduction in KV cache size vs standard MHA — while maintaining quality close to MHA. See `grouped_query_attention.md` for full treatment.

---

### Q7. How does Gemma differ from LLaMA?

**Answer.**

Gemma (Google DeepMind, 2024) shares much of the LLaMA recipe but introduces several refinements:

| Feature | LLaMA 2 | Gemma |
|---|---|---|
| Normalisation | RMSNorm (pre-norm) | RMSNorm (pre-norm) + post-norm |
| Positional encoding | RoPE | RoPE |
| Activation | SwiGLU | GeGLU |
| Attention | GQA (70B only) | MQA (2B), GQA (7B) |
| Vocabulary | 32,000 | 256,000 |
| Tokeniser | SentencePiece (BPE) | SentencePiece (BPE) |
| Context length | 4,096 | 8,192 |

Key Gemma-specific choices:

- **Post-MLP normalisation in addition to pre-norm**: applies RMSNorm after each sub-layer output (before adding the residual), providing additional training stability.
- **GeGLU**: uses GELU instead of Swish in the gated unit — $\text{GeGLU}(x) = \text{GELU}(xW) \otimes (xV)$
- **Large vocabulary (256k)**: improves multilingual and code tokenisation efficiency.
- **Logit soft-capping**: Gemma 2 applies $\tanh$ soft-capping to attention logits and final logits to improve training stability: $\tilde{a} = s \cdot \tanh(a / s)$ where $s$ is the cap value.

---

### Q8. Compare the architectural choices of GPT-2, LLaMA-2, Mistral 7B, and Gemma 7B.

**Answer.**

| Feature | GPT-2 | LLaMA-2 (7B) | Mistral 7B | Gemma 7B |
|---|---|---|---|---|
| Parameters | 1.5B (largest) | 7B | 7B | 7B |
| Layers | 48 | 32 | 32 | 28 |
| Hidden dim | 1,600 | 4,096 | 4,096 | 3,072 |
| Attention heads | 25 | 32 | 32 | 16 |
| KV heads | 32 (MHA) | 32 (MHA) | 8 (GQA) | 16 (MHA) |
| FFN type | MLP + GELU | SwiGLU | SwiGLU | GeGLU |
| Normalisation | Post-LayerNorm | Pre-RMSNorm | Pre-RMSNorm | Pre+Post RMSNorm |
| Position encoding | Learned absolute | RoPE | RoPE | RoPE |
| Context length | 1,024 | 4,096 | 8,192 (32k w/ SWA) | 8,192 |
| Tied embeddings | Yes | No | No | Yes |
| Biases | Yes | No | No | No |
| SWA | No | No | Yes ($W=4096$) | No |

---

## Advanced

### Q9. Why does pre-norm placement change the effective learning rate and gradient flow compared to post-norm?

**Answer.**

In **post-norm** (original Transformer): $\text{output} = \text{LN}(\mathbf{x} + F(\mathbf{x}))$

During backpropagation, gradients flow through the LayerNorm operation on the residual path. The normalisation can shrink or amplify gradients depending on the scale of activations, leading to gradient vanishing in very deep networks at initialisation (when the residual $F(\mathbf{x})$ is near zero).

In **pre-norm**: $\text{output} = \mathbf{x} + F(\text{LN}(\mathbf{x}))$

The residual connection $\mathbf{x}$ bypasses all normalisation, providing a direct gradient path from output to input:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_\ell} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\ell+1}} \cdot \left(1 + \frac{\partial F(\text{LN}(\mathbf{x}_\ell))}{\partial \mathbf{x}_\ell}\right)$$

The $1$ term ensures the gradient never vanishes through the skip path. This means pre-norm networks can be trained with larger learning rates and converge with less careful initialisation. The trade-off is that pre-norm can lead to representation degeneration at the final layer (the last layer's residuals are not normalised), which is why some models add a final LayerNorm after all blocks.

The practical consequence: post-norm models typically require careful learning rate warmup over thousands of steps; pre-norm models converge more robustly.

---

### Q10. How does weight tying between the embedding and unembedding matrices work, and why is it used?

**Answer.**

Weight tying (Press & Wolf, 2017) shares a single matrix $W_E \in \mathbb{R}^{V \times d}$ for both:
- **Input embedding**: converts token index $t$ to a dense vector — $\mathbf{e}_t = W_E[t]$ (row lookup)
- **Output projection (unembedding)**: converts hidden state to logits — $\text{logits} = h \cdot W_E^T$

**Why it works:** Both operations encode the same semantic information. The embedding matrix rows learn dense representations of tokens. The unembedding matrix rows learn what hidden-state "direction" corresponds to predicting each token. These are deeply related, and tying them encourages consistency.

**Memory saving:** For a vocabulary of $V = 50{,}257$ (GPT-2) and $d = 768$, the embedding matrix is $50{,}257 \times 768 \approx 154M$ parameters — a significant fraction of GPT-2 small's 117M total. Sharing this halves the memory for this component.

**When it is omitted:** LLaMA-2 and Mistral do not tie weights. At 7B+ scale, the relative memory saving is smaller, and untied weights give more representational freedom. Some analyses suggest untied models perform marginally better at large scale.

---

### Q11. Explain SwiGLU's relationship to the Gated Linear Unit family. Why does the gating mechanism help?

**Answer.**

Gated Linear Units (Dauphin et al., 2017) have the form:
$$\text{GLU}(\mathbf{x}, W, V) = (\mathbf{x}W) \otimes \sigma(\mathbf{x}V)$$

The sigmoid gate $\sigma(\mathbf{x}V) \in (0, 1)$ acts as a soft binary selector, allowing the model to suppress or amplify components of the linear projection $\mathbf{x}W$ conditioned on the input.

The GLU family generalises by replacing $\sigma$ with other activations:
- $\text{ReGLU}$: ReLU gate
- $\text{GEGLU}$: GELU gate (used in Gemma)
- $\text{SwiGLU}$: Swish (SiLU) gate — $\text{Swish}(z) = z \cdot \sigma(z)$

SwiGLU:
$$\text{SwiGLU}(\mathbf{x}) = (\mathbf{x} W_1) \otimes \text{Swish}(\mathbf{x} W_2)$$

The FFN block in LLaMA uses three matrices:
$$\text{FFN}(\mathbf{x}) = W_{\text{down}} \left[(\mathbf{x} W_{\text{up}}) \otimes \text{Swish}(\mathbf{x} W_{\text{gate}})\right]$$

**Why gating helps:** The multiplicative interaction creates a form of input-conditioned feature selection. Unlike a standard FFN where all neurons contribute to every output (additively), the gate can zero out irrelevant features. This provides a form of dynamic sparsity. Empirically, Shazeer (2020) showed GEGLU/SwiGLU consistently reduce perplexity by $\sim0.5\text{--}1$ bits/byte vs GELU FFN at matched parameter counts.

The parameter count cost: a standard $d \to 4d \to d$ FFN uses $2 \times 4d^2$ parameters. A SwiGLU FFN uses $3 \times \frac{8d^2}{3}$ parameters (with the $\frac{2}{3}$ scaling), keeping parameter count approximately equal.
