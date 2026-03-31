# Encoder-Decoder Structure

## Overview

The original Transformer (Vaswani et al., 2017) uses a symmetric encoder-decoder architecture. Understanding the role of each component — encoder stack, decoder stack, cross-attention, residual connections, layer normalisation, and feed-forward networks — and why each design choice was made is fundamental for ML interviews and for reasoning about modern architecture variants.

---

## Tier 1: Fundamentals

### Q1. Describe the high-level structure of the original Transformer. What does the encoder do and what does the decoder do?

**Answer.**

The original Transformer has two stacks:

**Encoder:**
- Takes the source sequence as input (e.g., a German sentence for translation)
- Produces a sequence of contextualised representations for each source token
- Uses bidirectional (non-causal) self-attention: every token can attend to every other token
- Does not generate output autoregressively; processes the entire source in parallel
- Stack of $N = 6$ identical layers (in the base model)

**Decoder:**
- Takes the (partially generated) target sequence and the encoder output
- Produces a probability distribution over the target vocabulary for the next token
- Uses causal self-attention (masked so no future tokens are visible)
- Uses cross-attention to query the encoder's output, conditioning generation on the source
- Stack of $N = 6$ identical layers

**Flow:**
```
Source tokens
     │
[Token Embeddings + Positional Encoding]
     │
┌────▼──────────────────────────┐
│  Encoder Layer ×6             │
│  ┌─ Self-Attention (bidir.)  │
│  └─ Feed-Forward Network     │
└────┬──────────────────────────┘
     │ Encoder output (all positions)
     │
     ╔══════════════════════════╗
     ║  Target tokens (shifted) ║
     ║  [Embeddings + PE]       ║
     ║  ┌─ Causal Self-Attn    ║
     ║  ├─ Cross-Attention ◄───╝
     ║  └─ Feed-Forward        ║
     ║  Decoder Layer ×6       ║
     ╚═══════════╤═════════════╝
                 │
           [Linear + Softmax]
                 │
           Next token probs
```

---

### Q2. What is cross-attention and how does it connect the encoder to the decoder?

**Answer.**

**Cross-attention** is an attention operation where:
- **Queries** ($Q$) come from the decoder's intermediate representations
- **Keys** ($K$) and **Values** ($V$) come from the encoder's final output

For each decoder position, cross-attention asks: "Given what I've generated so far (the query), which parts of the source sequence (keys/values) are most relevant?"

**Mechanically:**

Let $H_{\text{enc}} \in \mathbb{R}^{n_{\text{src}} \times d_{\text{model}}}$ be the encoder output and $H_{\text{dec}}^{(\ell-1)} \in \mathbb{R}^{n_{\text{tgt}} \times d_{\text{model}}}$ be the decoder's intermediate representation at layer $\ell$.

$$Q = H_{\text{dec}}^{(\ell-1)} W^Q, \quad K = H_{\text{enc}} W^K, \quad V = H_{\text{enc}} W^V$$

$$\text{CrossAttention} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The output has shape $n_{\text{tgt}} \times d_{\text{model}}$: one contextualised vector per decoder position, formed by attending over all encoder positions.

**Key property:** Cross-attention is **not masked** (unlike decoder self-attention). Every decoder position can attend to every encoder position — the decoder is allowed to look at the entire source.

**Why this design works:** The encoder has already produced rich, bidirectionally-contextualised representations. The cross-attention allows the decoder to access any part of this representation at each generation step, providing a flexible conditioning mechanism.

---

### Q3. What are residual connections and layer normalisation? Why are both used in the Transformer?

**Answer.**

**Residual connections (He et al., 2016):**

Each sub-layer in the Transformer (self-attention, cross-attention, FFN) is wrapped with a residual connection:

$$\text{output} = x + \text{Sublayer}(x)$$

where $x$ is the sub-layer input.

**Why residuals?**

1. **Gradient flow:** In deep networks, gradients must backpropagate through many layers. Without residuals, the gradient of the loss with respect to early layers involves products of many Jacobians, which can vanish or explode. Residuals create a "gradient highway": $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \text{output}} \cdot (1 + \frac{\partial \text{Sublayer}}{\partial x})$. The $+1$ ensures a non-vanishing gradient path.

2. **Initialisation near identity:** At initialisation, sub-layer outputs are near zero (small random weights). The residual connection means the initial forward pass is approximately the identity — a stable starting point.

3. **Depth enablement:** ResNets demonstrated that residuals allow training hundreds of layers; Transformers exploit the same property.

**Layer normalisation (Ba et al., 2016):**

Normalises activations across the feature dimension (not the batch dimension):

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

where $\mu = \text{mean}(x)$, $\sigma = \text{std}(x)$ computed over the $d_{\text{model}}$ dimensions, and $\gamma, \beta \in \mathbb{R}^{d_{\text{model}}}$ are learned scale and shift parameters.

**Why LayerNorm (not BatchNorm)?**

- BatchNorm normalises across the batch dimension — statistics depend on batch size. For variable-length sequences and small batches (common in NLP), BatchNorm is unstable.
- LayerNorm normalises within each token's feature vector — statistics are independent of batch size and sequence length.

**Pre-LN vs. Post-LN:**

The original Transformer uses **Post-LN**: $x + \text{LayerNorm}(\text{Sublayer}(x))$.

Modern practice uses **Pre-LN**: $x + \text{Sublayer}(\text{LayerNorm}(x))$.

Pre-LN is more training-stable: the gradient flows through the residual path without passing through normalisation, avoiding the vanishing gradient problem at depth. Most LLMs (GPT-2+, LLaMA, etc.) use Pre-LN.

---

### Q4. What is the feed-forward network (FFN) in a Transformer layer? What role does it play?

**Answer.**

Each Transformer layer contains a position-wise feed-forward network applied independently to each token:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

(using ReLU; original paper) or with GELU in modern practice.

**Dimensions:**
- Input: $x \in \mathbb{R}^{d_{\text{model}}}$
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, where $d_{ff} = 4 \times d_{\text{model}}$ typically
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$
- Output: $\in \mathbb{R}^{d_{\text{model}}}$

**Computational role:**

1. **Expand then contract:** $W_1$ projects to a higher-dimensional space ($4 \times d_{\text{model}}$); the non-linearity adds expressiveness; $W_2$ projects back. This is the same structure as the bottleneck in MLPs.

2. **Token-wise processing:** The FFN does not mix information across positions — it applies the same transformation to each token independently. This contrasts with attention, which is the position-mixing operation.

3. **Memory function:** Research (Geva et al., 2021) suggests that FFN layers act as key-value memories, with $W_1$ rows acting as keys that match input patterns and $W_2$ columns acting as values (output distributions). The FFN stores factual associations, while attention handles information routing.

**Parameter count:** $2 \times d_{\text{model}} \times d_{ff} = 2 \times d_{\text{model}} \times 4 d_{\text{model}} = 8 d_{\text{model}}^2$. For $d_{\text{model}} = 512$: 2,097,152 parameters — twice the parameter count of the MHA sub-layer.

---

### Q5. Why does the original Transformer use $N = 6$ layers? Is this principled?

**Answer.**

The choice of $N = 6$ layers in the base model is essentially **empirical and pragmatic**, not derived from first principles.

**The paper's justification:**

Vaswani et al. present results for a "base" model ($N = 6$, $d_{\text{model}} = 512$) and a "big" model ($N = 6$, $d_{\text{model}} = 1024$) — they kept depth constant and varied width. This was partly a compute-budget decision for the experiments.

**What ablations show:**

The original paper does include ablations varying $N$ from 2 to 6. Performance improves monotonically with depth up to 6 layers (in their compute budget), but the improvement plateaus. Within a fixed compute budget, the trade-off between depth and width (and attention heads) must be empirically determined.

**The broader principle:**

Scaling research (Kaplan et al., 2020) shows that increasing model depth (within a depth/width optimal ratio) consistently improves performance. Modern LLMs use $N = 32, 48, 80, 96$ layers. The "6 layers" choice was a practical compromise for 2017 hardware and the WMT translation task — not a fundamental architectural constant.

**What depth provides:**
- Each layer can build on the representations of the previous layer
- Early layers tend to capture local/surface features; later layers capture abstract semantic relationships
- Depth allows compositional computation that width alone cannot provide

---

## Tier 2: Intermediate

### Q6. Compare encoder-only, decoder-only, and encoder-decoder architectures. When is each appropriate?

**Answer.**

**Encoder-only (e.g., BERT, RoBERTa, DeBERTa):**

Architecture: Transformer encoder stack with bidirectional self-attention.

Pre-training: Masked language modelling (predict masked tokens) and optionally next sentence prediction.

Properties:
- Every token's representation sees the entire context (bidirectional)
- Not designed for autoregressive text generation
- Produces rich contextualised embeddings for each token

Best for:
- Classification (sentiment, NLI, topic)
- Named entity recognition (token-level classification)
- Question answering with extractive span selection
- Semantic similarity (encode both inputs, compare representations)
- Any discriminative task that doesn't require generating new text

**Decoder-only (e.g., GPT family, LLaMA, Mistral, Claude):**

Architecture: Transformer decoder stack with causal self-attention only (no cross-attention, no encoder).

Pre-training: Causal language modelling (predict the next token given the left context).

Properties:
- Each token can only attend to previous tokens
- Naturally suited for autoregressive generation
- KV caching enables efficient inference

Best for:
- Text generation (completion, chatbots, creative writing)
- In-context learning (few-shot prompting)
- Code generation
- Any generative task

**Encoder-decoder (e.g., original Transformer, T5, BART, Whisper):**

Architecture: Encoder stack + decoder stack with cross-attention.

Pre-training: T5 uses text-to-text format with span corruption; BART uses denoising autoencoding.

Properties:
- Encoder builds a rich bidirectional representation of the input
- Decoder generates output conditioned on the full encoded input
- More parameters at equal layer count than decoder-only (two stacks)

Best for:
- Machine translation (clear source → target structure)
- Summarisation
- Speech-to-text (Whisper)
- Document-level tasks where a complete source representation before generation helps

**Current trend:** Decoder-only models have come to dominate even tasks traditionally suited to encoder-decoder (translation, summarisation). Large decoder-only models generalise surprisingly well, and the simpler architecture simplifies training. However, encoder-decoder models remain competitive for tasks with clear input/output asymmetry and moderate context lengths.

---

### Q7. What is the "residual stream" interpretation of the Transformer? How does it change how we think about information flow?

**Answer.**

The **residual stream interpretation** (Elhage et al., 2021) views the Transformer as a sequence of read-and-write operations on a shared vector space for each token.

**Standard view:** Information flows through layers, being transformed at each step.

**Residual stream view:**

At each token position, there is a running vector $x^{(\ell)} \in \mathbb{R}^{d_{\text{model}}}$ that persists through all layers via residual connections:

$$x^{(\ell+1)} = x^{(\ell)} + \text{AttnOutput}^{(\ell)}(x^{(\ell)}) + \text{FFNOutput}^{(\ell)}(x^{(\ell)})$$

Each sub-layer **reads** from the stream and **writes back** an additive update. The stream accumulates contributions from all previous layers.

**What this view reveals:**

1. **Layers are additive:** Any layer's output is a delta added to the current representation. The final representation is literally the sum of all sub-layer outputs and the initial embedding.

2. **Attention heads write independently:** Each head's output is projected by a slice of $W^O$ and added to the stream. Different heads can write different information to different subspaces of $d_{\text{model}}$ without interference (if their write directions are orthogonal).

3. **Superposition:** Since $d_{\text{model}}$ is fixed but the model must represent many features, individual neurons are polysemantic — they participate in representing multiple unrelated features simultaneously. This has been empirically confirmed via sparse autoencoders.

4. **Circuit analysis:** We can trace how a specific piece of information (e.g., the subject of a sentence) gets written by an early attention head, persists in the residual stream, and gets read by a later head to compose a subject-verb agreement check.

5. **Layer skip connections are not "shortcuts":** They are the primary information pathway. The sub-layer computations are corrections/additions, not transformations of the main signal.

---

## Tier 3: Advanced

### Q8. Analyse the computational cost of a full Transformer forward pass. Where are the FLOPs concentrated?

**Answer.**

For an encoder-decoder Transformer with parameters $d_{\text{model}}$, $d_{ff} = 4d_{\text{model}}$, $N$ layers each, sequence length $n$, batch size $B = 1$:

**Per encoder layer:**

| Operation | FLOPs |
|---|---|
| Self-attention: $Q, K, V$ projections | $3 \times 2n d_{\text{model}}^2 = 6nd^2$ |
| Attention scores $QK^T$ | $2n^2 d_k \times h = 2n^2 d_{\text{model}}$ |
| Softmax | $O(n^2)$ (negligible vs. matmuls) |
| Weighted sum | $2n^2 d_{\text{model}}$ |
| Output projection $W^O$ | $2n d_{\text{model}}^2$ |
| FFN layer 1 | $2n d_{\text{model}} d_{ff} = 8nd^2$ |
| FFN layer 2 | $8nd^2$ |
| **Layer total** | $\approx (24nd^2 + 4n^2d)$ |

where $d = d_{\text{model}}$ for brevity.

**Total for $N$ encoder layers:**

$$\text{FLOPs}_{\text{enc}} \approx N(24nd^2 + 4n^2d)$$

**Regime analysis:**

The two terms are:
- $24nd^2$: scales with sequence length $n$ and model width $d^2$ — dominates for short sequences
- $4n^2d$: scales quadratically with sequence length — dominates for long sequences

**Crossover point:** $24nd^2 = 4n^2d \implies n = 6d$

For $d_{\text{model}} = 512$: crossover at $n = 3072$ tokens.
For $d_{\text{model}} = 12288$ (GPT-3): crossover at $n = 73,728$ tokens.

**Interpretation:** For most typical use cases (short-medium sequences with large $d_{\text{model}}$), the **FFN and projection matrices dominate**, not the attention. This is counterintuitive — we often think of attention as the expensive operation, but at large $d_{\text{model}}$, the linear layers dominate.

**Implication for efficiency:** Most parameter-efficient methods (LoRA, pruning heads) target the linear projections. Techniques like MoE (Mixture of Experts) replace the FFN with multiple sparse expert networks, targeting the largest parameter block.

---

### Q9. What modifications distinguish modern LLM architectures from the original 2017 Transformer? Discuss at least five substantive changes.

**Answer.**

Modern LLMs (LLaMA-3, Mistral, Gemma, GPT-4 class) differ from the 2017 Transformer in several important ways:

**1. Decoder-only architecture**

The encoder-decoder design was suited for sequence-to-sequence tasks. Modern LLMs are decoder-only: a single stack of causal self-attention layers. This simplifies training (one pretraining objective: next token prediction), and scaling laws have shown decoder-only models achieve strong performance across diverse tasks.

**2. Pre-Layer Norm (Pre-LN)**

Original Transformer: Post-LN ($\text{LayerNorm}$ applied after the residual addition).
Modern: Pre-LN ($\text{LayerNorm}$ applied to the input before the sub-layer, residual added after).

Pre-LN ensures that the residual path is unobstructed by normalisation, dramatically improving training stability for deep models.

**3. RMS Norm instead of Layer Norm**

LLaMA, Mistral, Gemma use RMSNorm (Zhang & Sennrich, 2019):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma$$

Removes the mean-centering step. Faster to compute (one less reduction), equally effective empirically.

**4. Rotary Position Embeddings (RoPE) or ALiBi**

Original: Fixed sinusoidal absolute PE added to input.
Modern: RoPE applied within each attention layer to Q and K. Better length generalisation; encodes relative position directly in the attention dot product.

**5. SwiGLU / GeGLU activation in FFN**

Original: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$

Modern (LLaMA, PaLM): SwiGLU (Shazeer, 2020):

$$\text{FFN}(x) = (\text{SiLU}(xW_1) \odot (xW_3)) W_2$$

Uses a gating mechanism: the SiLU-activated branch gates the linear branch element-wise. Requires three weight matrices instead of two (to maintain the same parameter count, $d_{ff}$ is reduced to $\frac{2}{3} \times 4d_{\text{model}} \approx 2.67d_{\text{model}}$). Empirically outperforms ReLU/GeLU across scales.

**6. Grouped-Query Attention (GQA)**

Original: Each head has independent Q, K, V projections (full MHA).
Modern (LLaMA-2 70B, LLaMA-3, Mistral): GQA with $g$ key-value head groups shared across query heads. Reduces KV cache memory and inference latency with minimal quality loss.

**7. No bias terms in linear layers**

Most modern LLMs remove bias terms from attention and FFN projections. Reduces parameters slightly; found to have negligible impact on performance. RMSNorm also has no bias.

**8. Tied input/output embeddings**

The token embedding matrix $W_e \in \mathbb{R}^{V \times d_{\text{model}}}$ is often shared with (transposed and used as) the final output projection $W_e^T \in \mathbb{R}^{d_{\text{model}} \times V}$. Reduces parameters by $V \times d_{\text{model}}$ (e.g., 100M parameters for $V = 50k$, $d = 2048$) at the cost of constraining the output projection.

**Summary table:**

| Component | 2017 Transformer | Modern LLM |
|---|---|---|
| Architecture | Encoder-decoder | Decoder-only |
| Normalisation position | Post-LN | Pre-LN |
| Norm type | LayerNorm | RMSNorm |
| Position encoding | Sinusoidal absolute | RoPE |
| FFN activation | ReLU | SwiGLU |
| Attention heads | Full MHA | GQA ($g < h$) |
| Biases | Yes | No (mostly) |
