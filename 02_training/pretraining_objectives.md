# Pretraining Objectives

## Overview

Pretraining objectives define what a model learns during its main training phase. The choice of objective fundamentally shapes what capabilities the model develops, how well it transfers to downstream tasks, and how efficiently it learns from data. Understanding the mechanics of causal LM, masked LM, and their variants — along with why autoregressive pretraining has come to dominate — is essential for ML research and engineering roles.

---

## Tier 1: Fundamentals

### Q1. What is causal language modelling (CLM)? Write the training objective formally.

**Answer.**

**Definition:** Causal language modelling (also called autoregressive language modelling, or GPT-style pretraining) trains the model to predict the next token given all preceding tokens in the sequence.

**Formal objective:**

Given a sequence of tokens $(x_1, x_2, \ldots, x_T)$, the model computes:

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

Using the chain rule of probability, this is equivalent to maximising:

$$p_\theta(x_1, \ldots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

**Implementation:**

During training, the entire sequence is processed in one forward pass using a **causal (lower-triangular) attention mask** that prevents position $t$ from attending to positions $t+1, \ldots, T$. This allows all $T$ next-token predictions to be computed in parallel, even though each prediction is conditioned only on preceding tokens.

**Key properties:**
- Fully unsupervised: labels are the input tokens shifted by one
- Every token in the sequence provides a training signal (100% token utilisation)
- The model is trained exactly as it will be used at inference: left-to-right generation
- The trained model is a proper probability distribution over sequences: $\sum_{\mathbf{x}} p_\theta(\mathbf{x}) = 1$

---

### Q2. What is masked language modelling (MLM)? How does BERT implement it?

**Answer.**

**Definition:** Masked language modelling trains the model to predict the identity of randomly masked tokens given the surrounding (bidirectional) context.

**BERT's implementation (Devlin et al., 2018):**

For each training sequence:

1. **Select 15% of token positions** uniformly at random
2. For each selected position:
   - 80% of the time: replace with the special `[MASK]` token
   - 10% of the time: replace with a random token
   - 10% of the time: leave unchanged

3. **Objective:** Predict the original token at each selected position:

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \text{masked}} \log p_\theta(x_t \mid \tilde{x}_1, \ldots, \tilde{x}_T)$$

where $\tilde{x}$ is the corrupted sequence and the sum is only over masked positions.

**Why the 80/10/10 split?**
- Pure 100% masking: the model learns to rely on `[MASK]` tokens, which don't appear at fine-tuning time (train-inference mismatch)
- 10% random: forces the model to build a contextual representation of every token, not just the masked ones
- 10% unchanged: forces the model to handle the case where the token is already correct

**Properties of MLM:**
- Bidirectional context: the model sees the full sequence (with masking)
- Only 15% of tokens contribute to the loss per step — less data efficient than CLM
- Not generative: you cannot sample new sequences directly from an MLM model
- Produces strong bidirectional representations suited for discriminative downstream tasks

---

### Q3. What is the Next Sentence Prediction (NSP) objective? Why was it later abandoned?

**Answer.**

**Definition (BERT):** In addition to MLM, BERT was trained to predict whether two sentences appear consecutively in the source document.

**Implementation:**
- 50% of training pairs: sentence B genuinely follows sentence A ("IsNext")
- 50% of training pairs: sentence B is a random sentence from a different document ("NotNext")
- A `[CLS]` token prepended to the sequence; its final representation is fed to a binary classifier

**Formal objective:**

$$\mathcal{L}_{\text{NSP}} = -[\mathbb{1}[\text{IsNext}]\log p_\theta(\text{IsNext}) + \mathbb{1}[\text{NotNext}]\log p_\theta(\text{NotNext})]$$

Total BERT loss: $\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$

**Why was it abandoned?**

**RoBERTa (Liu et al., 2019)** conducted ablation studies showing:
1. Removing NSP and training with longer sequences (full documents) consistently improves downstream performance
2. NSP is too easy: models can solve it using topic information alone (the random negative sentences are from different documents, hence different topics), without learning sentence-level coherence
3. Constructing NSP pairs disrupts the contiguous document structure that is more informative for MLM

**Successor objectives:**
- **Sentence Order Prediction (SOP, ALBERT):** Predict whether two consecutive segments are in the correct order vs. swapped. Harder (both segments come from the same document) and more useful.
- **Most modern models:** Drop sentence-level objectives entirely and use only MLM or CLM on long contiguous sequences.

---

### Q4. What is a Prefix LM objective? How does it differ from both CLM and MLM?

**Answer.**

A **Prefix LM** (also called semi-causal or bidirectional-causal) divides each sequence into a prefix and a continuation:

- The **prefix** (first $k$ tokens): bidirectional attention — each prefix token can attend to all other prefix tokens
- The **continuation** (remaining tokens): causal attention — each token attends to all prefix tokens and previous continuation tokens

**Formal objective:** Compute loss only on the continuation tokens:

$$\mathcal{L}_{\text{PrefixLM}} = -\sum_{t=k+1}^{T} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

The prefix serves as conditioning context with full bidirectional processing.

**Used in:** T5 (sort of — via the encoder-decoder formulation), PaLM, and the original "Prefix LM" framing in Raffel et al. (2020).

**Comparison:**

| Property | CLM | MLM | Prefix LM |
|---|---|---|---|
| Context direction | Left-only (causal) | Bidirectional | Prefix: bidir.; continuation: causal |
| Generative | Yes | No | Yes |
| Training signal | All tokens | 15% of tokens | Continuation tokens only |
| Architecture | Decoder-only | Encoder-only | Can be decoder-only with partial mask |
| Best for | Generation, ICL | Classification, understanding | Conditional generation |

---

## Tier 2: Intermediate

### Q5. Why has autoregressive (causal LM) pretraining come to dominate for large language models? Analyse the trade-offs.

**Answer.**

**Argument 1: Generality via the chain rule**

The CLM objective directly maximises $p(x_1, \ldots, x_T) = \prod_t p(x_t \mid x_{<t})$ — a complete probability model of the sequence. This model can, in principle, do anything that a language model needs to do:

- Unconditional generation: sample from $p(x)$
- Conditional generation: provide a prefix and sample the continuation
- Discriminative tasks (via prompting): compute $p(\text{label} \mid \text{input})$ and choose the highest-probability label
- In-context learning: few-shot examples in the prompt condition the generation

MLM models cannot generate new text and require task-specific fine-tuning heads. They are inherently discriminative.

**Argument 2: Scale efficiency**

Every token in a CLM training batch contributes to the loss (100% utilisation). MLM uses only ~15% of tokens per sequence for the gradient signal. At scale, this translates to a meaningful efficiency difference.

**Argument 3: Emerging capabilities at scale**

GPT-3 demonstrated that a pure CLM model, trained at sufficient scale, develops strong few-shot learning from the pretraining distribution alone. Encoder-only models plateau: their capability is largely capped by the quality of the fine-tuning data, not the scale of pretraining.

**Argument 4: Simpler training and deployment**

Single objective, single model architecture, no special tokens or corruption procedure. At inference, the same model used for pretraining is used directly (with prompting or fine-tuning).

**The case for MLM/encoder-only:**

- For classification, ranking, and embedding-intensive tasks (semantic search, NLI), encoder-only models trained with MLM can outperform similarly-sized CLM models
- MLM produces better raw token representations for dense retrieval
- Encoder models are cheaper to run for embedding tasks (one forward pass, no generation)

**Current consensus:** For frontier AI, decoder-only CLM models dominate because of their generality and scaling properties. Encoder-only models remain useful for retrieval, classification, and efficiency-constrained deployments.

---

### Q6. What are the practical differences between training on documents vs. sentence pairs? How does data packing affect the training objective?

**Answer.**

**Document-level training (RoBERTa, GPT-2+, LLaMA):**

Pack full documents (or large chunks) as training sequences. When a sequence budget is filled, a new document begins (possibly with a separator token).

- Preserves long-range coherence within documents
- The model can learn cross-sentence dependencies naturally
- Document boundaries are marked so the model doesn't attend across them (or a separator token signals the boundary)

**Sentence pair training (original BERT):**

Training sequences consist of two sentences sampled from the document corpus, concatenated with `[SEP]`. Specifically:

- 50% of cases: consecutive sentences from the same document
- 50% of cases: a sentence paired with a random sentence from another document

- Artificial constraint on context length (each sentence is short)
- Leads to information loss compared to full document packing
- Was motivated by the NSP objective, which has since been deprecated

**Data packing (concatenation + batching):**

Modern practice concatenates multiple documents into one long sequence up to the model's context length, inserting a special end-of-document token `<|endoftext|>`. This maximises GPU utilisation — no padding.

**Cross-document attention issue:** If tokens from two different documents are in the same sequence without masking at document boundaries, the model can attend across documents. This is typically handled by:
1. Adding `<|endoftext|>` tokens and training the model to not attend across them
2. Using document-level attention masks (more complex)
3. Accepting minor cross-document contamination (common in practice for large-scale training)

**Impact on training objective:** When packing multiple short documents, the average sequence has fewer natural continuation patterns from a single document. This slightly degrades performance on tasks requiring very long-range context but is generally accepted for the efficiency gains.

---

### Q7. What is the perplexity metric and how does it relate to the CLM training objective?

**Answer.**

**Perplexity** is the standard evaluation metric for language models. For a sequence $(x_1, \ldots, x_T)$:

$$\text{PPL} = \exp\!\left(\frac{1}{T} \sum_{t=1}^{T} -\log p_\theta(x_t \mid x_{<t})\right) = \exp\!\left(\frac{\mathcal{L}_{\text{CLM}}}{T}\right)$$

It is the exponential of the per-token cross-entropy loss, normalised by sequence length.

**Intuition:**

Perplexity measures how "surprised" the model is by each new token on average. A perplexity of $k$ means the model is as uncertain as if it had to choose uniformly among $k$ options at each step.

- $\text{PPL} = 1$: perfect prediction (model is never surprised)
- $\text{PPL} = V$: uniform distribution over vocabulary (random guessing)
- $\text{PPL} \approx 20$: approximate perplexity of a strong LLM on standard benchmarks (e.g., WikiText-103)

**Relationship to training loss:**

The CLM training loss is directly the per-token negative log-likelihood:

$$\mathcal{L} = -\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t}) = \log(\text{PPL})$$

Minimising training loss is equivalent to minimising perplexity. Lower perplexity = better compression of the data distribution = better model.

**Important caveats:**

1. **Tokenisation dependence:** Perplexity values are only comparable across models with the same tokenisation. A model with a larger vocabulary may have lower perplexity simply because it breaks text into fewer tokens.

2. **Domain dependence:** Perplexity on Wikipedia text tells you nothing about perplexity on code or medical text. Models should be evaluated on perplexity on held-out data from their target domain.

3. **PPL is not downstream task performance.** A model with lower perplexity doesn't always perform better on reasoning tasks. Capability elicitation (prompting, fine-tuning) matters enormously.

---

## Tier 3: Advanced

### Q8. Compare the data efficiency of CLM, MLM, and span corruption (T5-style). Which provides the most bits of information per training example?

**Answer.**

**CLM (next token prediction):**

For a sequence of $T$ tokens:
- Loss signal: $T$ next-token predictions
- Each prediction: $\log_2 V$ bits theoretically; in practice $H(x_t \mid x_{<t})$ bits per token
- Total information: proportional to $T \cdot H$ where $H$ is the average per-token entropy

**MLM (masked language modelling):**

- Mask fraction $m = 0.15$, so $mT$ tokens contribute to loss
- Each masked prediction sees full bidirectional context, so the conditional entropy is lower: $H_{\text{bidir}}(x_t \mid x_{\neq t}) \leq H(x_t \mid x_{<t})$
- Total information: $mT \cdot H_{\text{bidir}} \approx 0.15T \cdot H_{\text{bidir}}$

Since $H_{\text{bidir}} < H_{\text{causal}}$ (bidirectional context is more informative, leaving less uncertainty), and $m = 0.15$, MLM provides significantly less gradient signal per sequence:

$$\text{Info}_{\text{MLM}} < 0.15 \times \text{Info}_{\text{CLM}}$$

**Span corruption (T5-style):**

T5 masks contiguous spans averaging 3 tokens at a 15% token rate. The decoder predicts the masked spans autoregressively.

- First masked token in a span: moderate uncertainty (knows context from both sides, but not the span content)
- Subsequent tokens in span: easier (given the beginning of the span, each continuation is more predictable)
- Overhead: sentinel tokens add non-content tokens to the vocabulary predictions

T5's span corruption is more efficient than per-token masking because consecutive predictions within a span have higher entropy (no bidirectional context for the span itself), but it still uses only 15% of tokens.

**Empirical comparison:**

RoBERTa (MLM) vs GPT-2 (CLM) with equal compute: on generation tasks, CLM wins; on classification tasks, MLM is competitive or better for smaller scales. At 10B+ parameters, CLM (in-context learning) matches or exceeds MLM + fine-tuning for most tasks, suggesting CLM becomes more data-efficient at scale due to better generalisation from the causal objective.

**The key insight:** CLM achieves full 100% token utilisation at the cost of a harder (more constrained) prediction problem (left-context only). MLM achieves easier predictions with richer context but wastes 85% of the potential training signal. For the same compute, CLM is the better pretraining objective for general capabilities.

---

### Q9. What is "teacher forcing" in the context of CLM training, and what is the exposure bias problem it creates?

**Answer.**

**Teacher forcing:**

During CLM training, the model receives the **ground-truth previous tokens** as input at each position, regardless of what it would have predicted. At step $t$, the input is always the true token $x_{t-1}$, not the model's predicted token $\hat{x}_{t-1}$.

This is what allows the entire sequence to be processed in one forward pass using the causal mask — every position receives its true prefix.

**The exposure bias problem (Bengio et al., 2015):**

At inference time, the model generates autoregressively: the predicted token $\hat{x}_{t-1}$ is fed as input for step $t$. If $\hat{x}_{t-1} \neq x_{t-1}$, the model is now in a state it has never encountered during training — the distribution of inputs at inference diverges from the distribution seen during training.

This creates a **compounding error** problem: early mistakes shift the generation into out-of-distribution states, which increases the probability of further mistakes, leading to degenerate outputs for long sequences.

**Manifestations:**
- Repetition loops in generation (the model gets stuck in a local region of the distribution)
- Hallucinations (reasoning errors compound into factually wrong outputs)
- Quality degradation for very long generations

**Mitigation approaches:**

1. **Scheduled Sampling (Bengio et al., 2015):** During training, with probability $p$ (decaying over training), use the model's own prediction instead of the ground truth. This exposes the model to its own distribution. Difficult to implement for Transformers (breaks parallelism).

2. **Reinforcement learning fine-tuning (RLHF/PPO, REINFORCE):** Fine-tune on the model's own generated outputs with a reward signal. This directly addresses the train-inference distribution mismatch.

3. **Minimum Bayes Risk (MBR) decoding:** At inference, generate many candidates and select the one with the highest expected quality — reduces sensitivity to individual token errors.

4. **Temperature/sampling strategies:** Nucleus sampling, top-$k$ sampling, and beam search are inference-time mitigations that reduce the probability of catastrophic single-step errors.

**Practical significance:** For most use cases (generation up to a few hundred tokens), exposure bias is manageable and teacher forcing is the standard. The issue becomes more severe for very long generations, mathematical reasoning chains, and code generation, which is part of the motivation for RLHF and reasoning-focused fine-tuning approaches.
