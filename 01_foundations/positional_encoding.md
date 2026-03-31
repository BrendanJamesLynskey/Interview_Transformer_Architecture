# Positional Encoding

## Overview

Because attention is permutation-equivariant, Transformers require explicit position information to distinguish token order. Positional encoding is the mechanism that injects this information. The design space ranges from the original sinusoidal fixed encodings to learned embeddings, relative encodings, rotary embeddings, and ALiBi. Understanding why each approach works and its trade-offs is frequently tested in ML research and engineering interviews.

---

## Tier 1: Fundamentals

### Q1. Why do Transformers need positional encodings? What happens if you omit them?

**Answer.**

**The permutation equivariance problem:**

Scaled dot-product attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The score between token $i$ and token $j$ depends only on $q_i \cdot k_j$ — a function of the token feature vectors alone, with no dependence on the positions $i$ or $j$. If you permute the input sequence by any permutation $\pi$, the output is permuted by the same $\pi$, but the model cannot distinguish the original order from the permuted one.

**Formally:** For any permutation matrix $P$, $\text{Attention}(PQ, PK, PV) = P \cdot \text{Attention}(Q, K, V)$.

**What happens without positional encodings:**

- The model treats the sentence as a bag of tokens with no order
- "The cat sat on the mat" and "The mat sat on the cat" produce identical representations
- For language modelling, the model cannot learn that word order carries meaning
- Empirically, models trained without any positional encoding perform drastically worse on tasks requiring sequential reasoning

**Positional encodings break this symmetry** by adding position-dependent information to the token embeddings before (or during) attention computation, making each position distinguishable.

---

### Q2. Describe the sinusoidal positional encoding used in the original Transformer. What are its properties?

**Answer.**

**Definition:**

For position $\text{pos}$ (integer, 0-indexed) and dimension index $i$ (0-indexed):

$$PE_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

The encoding is a $d_{\text{model}}$-dimensional vector for each position. Even-indexed dimensions use sine; odd-indexed use cosine.

**Key properties:**

**1. Bounded:** $PE_{(\text{pos},j)} \in [-1, 1]$ for all positions and dimensions. The encoding doesn't grow with position, so it doesn't dominate the token embedding.

**2. Unique:** Each position receives a unique encoding vector. The frequency $1 / 10000^{2i/d_{\text{model}}}$ ranges from $1$ (fastest oscillation, small $i$) to $1/10000$ (slowest oscillation, large $i$). This is analogous to a binary counter: low-order bits flip fast, high-order bits flip slowly.

**3. Relative positions are representable as linear functions:** For a fixed offset $k$, $PE_{\text{pos}+k}$ can be written as a linear function of $PE_{\text{pos}}$:

$$\begin{pmatrix} \sin((\text{pos}+k)\omega) \\ \cos((\text{pos}+k)\omega) \end{pmatrix} = \begin{pmatrix} \cos(k\omega) & \sin(k\omega) \\ -\sin(k\omega) & \cos(k\omega) \end{pmatrix} \begin{pmatrix} \sin(\text{pos}\cdot\omega) \\ \cos(\text{pos}\cdot\omega) \end{pmatrix}$$

where $\omega = 1/10000^{2i/d_{\text{model}}}$. This means the Transformer can, in principle, learn to extract relative position information via linear operations over the encoding.

**4. Generalisable to unseen lengths:** Since the encoding is a deterministic function, it naturally extends to any sequence length — the model can in principle generalise to longer sequences than seen during training.

---

### Q3. What is the difference between absolute and relative positional encodings?

**Answer.**

**Absolute positional encodings:**

Each token at position $p$ receives an encoding $PE(p)$ that encodes its absolute position in the sequence. The original sinusoidal PE and most learned PE approaches are absolute.

- Addition to input: $\tilde{x}_p = x_p + PE(p)$ before attention layers
- Pros: Simple, no architectural changes required
- Cons: The model must learn to infer relative distances from absolute positions; generalisation to longer sequences than those seen in training is unreliable

**Relative positional encodings:**

The encoding captures the offset $r = j - i$ between two tokens $i$ and $j$ rather than their absolute positions.

- Shaw et al. (2018): Add relative position biases to the attention scores $e_{ij} \to e_{ij} + a_{r}^K$ and optionally to the value aggregation
- The attention score between $i$ and $j$ becomes a function of both token content and relative distance

$$e_{ij} = \frac{(x_i W^Q)(x_j W^K + a_{ij}^K)^T}{\sqrt{d_k}}$$

where $a_{ij}^K = w_{\text{clip}(j-i, -k, k)}$ is a learned embedding for each relative offset.

**Key distinction:**

| | Absolute | Relative |
|---|---|---|
| What is encoded | Position $p$ of a token | Distance $j - i$ between two tokens |
| Location of encoding | Added to input | Added to attention scores or values |
| Length generalisation | Poor (at exact boundary) | Better for moderate extensions |
| Examples | Sinusoidal PE, learned PE | Shaw et al., T5 biases, RoPE, ALiBi |

---

### Q4. What are learned positional embeddings? How do they compare to sinusoidal encodings?

**Answer.**

**Learned positional embeddings:**

Instead of a fixed formula, define a lookup table:

$$PE \in \mathbb{R}^{L_{\text{max}} \times d_{\text{model}}}$$

where $L_{\text{max}}$ is the maximum supported sequence length. Row $p$ is the learned embedding for position $p$, initialised randomly and trained end-to-end with the model.

**Used in:** BERT, GPT-2, many standard Transformer implementations.

**Comparison:**

| Property | Sinusoidal (fixed) | Learned |
|---|---|---|
| Parameters | 0 (no learnable params) | $L_{\text{max}} \times d_{\text{model}}$ |
| Length extrapolation | Degrades gracefully | Fails hard (no embeddings beyond $L_{\text{max}}$) |
| Flexibility | Fixed frequency structure | Can adapt to data distribution |
| Interpretability | Known mathematical properties | Opaque |
| Performance | Comparable | Slightly better on some benchmarks |

**Vaswani et al. (2017) finding:** In their ablation, sinusoidal and learned positional embeddings produced nearly identical results. This suggests the architecture is robust to the specific form of positional encoding, as long as position information is provided.

**Practical implication:** Learned embeddings are preferred when you know the sequence length at train time. Sinusoidal or relative encodings are preferred when length generalisation matters.

---

## Tier 2: Intermediate

### Q5. Prove the linear representability of relative positions under sinusoidal encoding. Derive the rotation matrix explicitly.

**Answer.**

**Claim:** For each frequency $\omega_i$, there exists a $2 \times 2$ matrix $M_k^{(i)}$ such that:

$$\begin{pmatrix} PE_{(\text{pos}+k,\, 2i)} \\ PE_{(\text{pos}+k,\, 2i+1)} \end{pmatrix} = M_k^{(i)} \begin{pmatrix} PE_{(\text{pos},\, 2i)} \\ PE_{(\text{pos},\, 2i+1)} \end{pmatrix}$$

**Derivation:**

Let $\omega = \omega_i = 1/10000^{2i/d_{\text{model}}}$ and let $\phi = \text{pos} \cdot \omega$.

$$PE_{(\text{pos},\, 2i)} = \sin(\phi), \quad PE_{(\text{pos},\, 2i+1)} = \cos(\phi)$$

For position $\text{pos} + k$, angle is $(\text{pos} + k)\omega = \phi + k\omega$.

Using the angle addition identities:

$$\sin(\phi + k\omega) = \sin(\phi)\cos(k\omega) + \cos(\phi)\sin(k\omega)$$
$$\cos(\phi + k\omega) = \cos(\phi)\cos(k\omega) - \sin(\phi)\sin(k\omega)$$

In matrix form:

$$\begin{pmatrix} \sin(\phi + k\omega) \\ \cos(\phi + k\omega) \end{pmatrix} = \underbrace{\begin{pmatrix} \cos(k\omega) & \sin(k\omega) \\ -\sin(k\omega) & \cos(k\omega) \end{pmatrix}}_{M_k^{(i)}} \begin{pmatrix} \sin(\phi) \\ \cos(\phi) \end{pmatrix}$$

**Key observation:** $M_k^{(i)}$ is a **rotation matrix** by angle $-k\omega_i$. It depends only on $k$ (the relative offset) and $\omega_i$ (the frequency for this pair of dimensions), not on the absolute position $\text{pos}$.

This means the full sinusoidal PE satisfies:

$$PE_{\text{pos}+k} = \text{BlockDiag}(M_k^{(0)}, M_k^{(1)}, \ldots, M_k^{(d/2-1)}) \cdot PE_{\text{pos}}$$

The transformation from $PE_{\text{pos}}$ to $PE_{\text{pos}+k}$ is a fixed linear map (block-diagonal rotation matrix) parameterised only by $k$. Therefore, an attention layer could in principle implement relative position awareness by learning this linear map.

**Why this matters for the model:**

The $W^Q$ and $W^K$ projection matrices can, in combination with these linear transformations, extract relative position signals from absolute encodings. This is a theoretical justification for why absolute sinusoidal encoding supports position-relative reasoning, even though no relative position terms appear explicitly.

---

### Q6. Compare ALiBi, RoPE, and T5 relative biases as positional encoding strategies. What are the trade-offs for long-context applications?

**Answer.**

**T5 relative position biases (Raffel et al., 2020):**

Learned scalar biases added to attention scores based on relative offset:

$$e_{ij} \leftarrow e_{ij} + b_{f(i-j)}$$

where $f$ is a bucketing function (buckets at distance 1, 2, 4, 8, ...) and $b$ is a learned embedding. The bucketing means far-away positions share a bias.

- Pros: Expressive, learned; used in T5 and its variants
- Cons: Additional parameters per layer per head; bucketing loses fine-grained distance information; poor zero-shot length extrapolation beyond training range

**ALiBi — Attention with Linear Biases (Press et al., 2021):**

A fixed (non-learned) linear penalty on attention scores based on distance:

$$e_{ij} \leftarrow e_{ij} - m_h \cdot |i - j|$$

where $m_h$ is a fixed head-specific slope ($m_h = 2^{-8h/n_{\text{heads}}}$ for the $h$-th head).

- Pros: No parameters; trains at short context, extrapolates to longer contexts (demonstrated: train at 1k, infer at 2k with quality retention)
- Cons: The linear penalty biases the model towards local attention; may hurt tasks requiring long-range retrieval; slope schedule is a fixed heuristic

**RoPE — Rotary Position Embeddings (Su et al., 2021):**

Applied to Q and K via position-dependent rotation (see `multi_head_attention.md` for derivation). The dot product inherently encodes relative position.

- Pros: Elegant relative encoding property; proven extrapolation with extensions (YaRN, LongRoPE); dominant in modern open models
- Cons: Extrapolation degrades without modifications; requires applying rotations to every Q/K at every layer

**Comparison for long-context use:**

| Method | Type | Length extrapolation | Parameters added | Models |
|---|---|---|---|---|
| Sinusoidal | Absolute (fixed) | Poor | 0 | Original Transformer |
| Learned | Absolute (learned) | Fails hard | $L_{\text{max}} \times d$ | BERT, GPT-2 |
| T5 biases | Relative (learned) | Moderate | Per-layer-head | T5, mT5 |
| ALiBi | Relative (fixed) | Good (linear) | 0 | BLOOM, MPT |
| RoPE | Relative (rotary) | Good (+extensions) | 0 | LLaMA, Mistral, Gemma |

**Current best practice (2025):** RoPE with YaRN or LongRoPE for contexts beyond the training length is the dominant choice. ALiBi remains interesting for streaming/unlimited-length inference scenarios.

---

## Tier 3: Advanced

### Q7. Derive the YaRN method for extending RoPE to longer sequences. What modification does it make to the base RoPE frequencies?

**Answer.**

**The problem with vanilla RoPE at long contexts:**

RoPE assigns each dimension pair $(2i, 2i+1)$ a frequency $\theta_i = 10000^{-2i/d_k}$. For position $m$, the rotation angle is $m \cdot \theta_i$.

At training context length $L$, the maximum angle for the fastest-varying dimensions is $L \cdot \theta_0 = L \cdot 1 = L$ radians. The model sees full rotation cycles for all dimensions.

When extending to $L' > L$ (e.g., $L = 4096$, $L' = 32768$), the angles exceed what the model was trained on. High-frequency dimensions (small $i$) cycle many times in ways unseen during training, causing the model to treat distant tokens as if they were close (aliasing). Empirically, perplexity spikes dramatically beyond the training context.

**Position interpolation (Chen et al., 2023):**

A simple fix: scale all positions by $L/L'$ so that the effective context is always $\leq L$:

$$m \leftarrow m \cdot \frac{L}{L'}$$

This works but requires fine-tuning and degrades local attention precision (nearby tokens that differ by small positions now have positions squashed together).

**YaRN — Yet Another RoPE Extension (Peng et al., 2023):**

YaRN applies a non-uniform frequency scaling. Instead of scaling all frequencies equally, it:

1. **Partitions** dimensions into three groups based on frequency:
   - High-frequency dimensions ($\theta_i$ large): leave unchanged (no interpolation needed)
   - Low-frequency dimensions ($\theta_i$ small): apply full interpolation
   - Mid-frequency: apply partial interpolation

2. **Ramps** the interpolation factor smoothly as a function of $i$:

$$\hat{\theta}_i = \begin{cases} \theta_i & \text{if } \lambda_i \geq \lambda_{\text{high}} \\ \theta_i / s & \text{if } \lambda_i \leq \lambda_{\text{low}} \\ \theta_i \left(\frac{1-\alpha}{\lambda_i} + \alpha\right)^{-1} & \text{otherwise} \end{cases}$$

where $\lambda_i = 2\pi / \theta_i$ is the wavelength, $s = L'/L$ is the extension ratio, and $\alpha, \lambda_{\text{high}}, \lambda_{\text{low}}$ are hyperparameters.

3. **Applies a temperature factor** to attention (analogous to reducing temperature $\sqrt{d_k}$) to counteract the broadening of the attention distribution at long contexts.

**Why this works better than uniform interpolation:**

- High-frequency dimensions handle local context. Squashing them together destroys local precision.
- Low-frequency dimensions handle long-range context. These need interpolation for long sequences.
- YaRN preserves local attention quality while enabling extension, requiring only minimal fine-tuning (a few hundred steps on long documents).

**Practical result:** Mistral 7B extended from 8k to 128k context using YaRN, Llama-3 uses a variant called RoPE scaling with adjusted base frequency, and LongRoPE (Ding et al., 2024) identifies non-uniform dimension-specific scales empirically.

---

### Q8. Design an experiment to test whether a model has truly learned to use positional information vs. pattern-matching on lexical cues. What would good vs. poor positional generalisation look like?

**Answer.**

**The confound:** Many NLP benchmarks can be solved with lexical co-occurrence statistics without understanding position. A model might appear to use positional information but is actually exploiting surface patterns.

**Proposed experimental design:**

**Task: Positional Probe**

1. **Construct synthetic data:** Generate sentences where the answer depends strictly on position, not content. Example:
   - "A B C D E" → "What is the token at position 3?" → "C"
   - Vary token vocabulary uniformly to eliminate lexical shortcuts

2. **Control conditions:**
   - Random shuffles of the same tokens: "C A E B D" → same question → should answer "B"
   - If the model answers correctly, it uses true positional encoding
   - If it answers "C" regardless of shuffle, it is exploiting token identity, not position

3. **Out-of-distribution length test:**
   - Train on sequences of length $\leq 20$
   - Test on sequences of length 50, 100, 200
   - Measure accuracy degradation by position scheme

**Metrics:**

| Condition | Good positional generalisation | Poor positional generalisation |
|---|---|---|
| In-distribution positions | Near-100% accuracy | Near-100% accuracy |
| Shuffled tokens | Correct (adapts to new order) | Same answer as unshuffled |
| OOD positions (2x train length) | Graceful degradation | Sharp accuracy cliff |
| OOD positions (8x train length) | Moderate degradation | Near-random |

**What to expect from each PE type:**

- **Sinusoidal (absolute):** Fails badly at 4x+ training length; moderate degradation within 2x
- **Learned (absolute):** Hard failure beyond $L_{\text{max}}$ (undefined embeddings)
- **ALiBi:** Maintains accuracy at 2-3x training length; degrades more slowly
- **RoPE:** Degrades around 2-4x without modification; YaRN/LongRoPE extend this to 8-16x
- **RoPE + long-context fine-tuning:** Best OOD generalisation

**Additional diagnostic:** Attention pattern visualisation at long context. A model with good positional encoding should show sharply peaked local attention for positions near the training boundary; a model struggling with length extrapolation will show diffuse or incoherent patterns at positions far beyond training range.
