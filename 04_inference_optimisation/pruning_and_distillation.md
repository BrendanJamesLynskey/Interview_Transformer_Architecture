# Pruning and Knowledge Distillation

## Overview

Pruning and distillation are complementary model compression techniques. Pruning removes parameters from an existing model; distillation trains a smaller model to mimic a larger one. Both aim to reduce inference cost while retaining as much quality as possible.

---

## Fundamentals

### Q1. What is the difference between unstructured and structured pruning?

**Answer.**

**Unstructured pruning** sets individual weights to zero, regardless of their position in the weight matrix. The sparsity pattern is irregular.

Example: for $W \in \mathbb{R}^{4 \times 4}$, set the $40\%$ of weights with smallest absolute value to zero:

$$W = \begin{bmatrix} 0.8 & 0.1 & 0.5 & 0.2 \\ 0.3 & 0.9 & 0.1 & 0.7 \\ 0.2 & 0.4 & 0.8 & 0.1 \\ 0.6 & 0.1 & 0.3 & 0.5 \end{bmatrix} \to \hat{W} = \begin{bmatrix} 0.8 & 0 & 0.5 & 0 \\ 0.3 & 0.9 & 0 & 0.7 \\ 0 & 0.4 & 0.8 & 0 \\ 0.6 & 0 & 0 & 0.5 \end{bmatrix}$$

**Structured pruning** removes entire structural units: heads, layers, neurons, or channels. The result is a smaller but dense matrix.

Example: remove attention heads 3 and 7 from a 16-head attention layer — the new layer has 14 heads, but no sparse operations are needed.

**Key trade-off:**

| Property | Unstructured | Structured |
|---|---|---|
| Accuracy at high sparsity | Better (finer granularity) | Worse (coarser) |
| Inference speedup | Requires sparse hardware support | Works on any hardware |
| Implementation complexity | High (sparse kernels needed) | Low (standard dense ops) |
| Memory saving | Needs sparse storage format | Immediate (smaller matrices) |

**Practical implication.** Unstructured pruning theoretically can remove $90\%+$ of weights with modest quality loss, but standard GPU hardware runs dense matrix multiplications — sparse operations are not faster unless sparsity exceeds $\sim 50\text{--}80\%$ and specialised sparse kernels are used. Structured pruning is almost always preferred for LLM deployment.

---

### Q2. What is magnitude pruning?

**Answer.**

Magnitude pruning is the simplest pruning criterion: rank weights by their absolute value and remove those below a threshold.

**Global magnitude pruning:**
1. Compute $|w_{ij}|$ for all weights
2. Sort globally across all layers
3. Set the bottom $p\%$ to zero (where $p$ is the target sparsity)

**Layer-wise magnitude pruning:**
1. For each layer independently, remove the bottom $p\%$ of that layer's weights
2. Allows different effective sparsity per layer while targeting a uniform ratio

**Sensitivity analysis.** Not all layers are equally sensitive to pruning. Early layers (input embeddings, first few transformer layers) are typically more sensitive than middle layers. The output projection (language model head) is also sensitive. Common practice: apply lower sparsity to sensitive layers and higher to less sensitive ones.

**Iterative magnitude pruning.** One-shot pruning at high sparsity degrades quality significantly. Iterative pruning alternates between pruning a small fraction and fine-tuning:

```
for step in 1..K:
    prune bottom (p/K)% of remaining weights globally
    fine-tune for a few thousand steps
```

This "gradual pruning" schedule recovers most of the quality that one-shot pruning loses.

---

### Q3. What is knowledge distillation?

**Answer.**

Knowledge distillation (Hinton et al., 2015) trains a small **student model** to mimic a larger **teacher model** by learning from the teacher's output distribution, not just the ground-truth labels.

**Standard distillation loss:**

$$\mathcal{L}_{\text{distill}} = (1 - \lambda) \mathcal{L}_{\text{CE}}(y_{\text{true}}, \hat{y}_s) + \lambda \mathcal{L}_{\text{KL}}(p_T, p_S)$$

where:
- $\hat{y}_s$ is the student's predicted distribution
- $p_T = \text{Softmax}(z_T / \tau)$ is the teacher's softened distribution (logits divided by temperature $\tau > 1$)
- $p_S = \text{Softmax}(z_S / \tau)$ is the student's softened distribution
- $\mathcal{L}_{\text{KL}}(p_T, p_S) = \sum_v p_T(v) \log\frac{p_T(v)}{p_S(v)}$ is the KL divergence
- $\lambda$ balances student accuracy vs teacher imitation

**Why soft labels help.** The teacher's soft distribution encodes more information than the one-hot hard label. For example, if the teacher assigns probabilities $[0.85, 0.10, 0.03, 0.02]$ to $[\text{"cat"}, \text{"dog"}, \text{"tiger"}, \text{"fox"}]$, the student learns that "cat" is most likely but "dog" is a plausible alternative — and that "cat" is more similar to "dog" than to "fox". This **dark knowledge** (the relative probabilities of incorrect classes) is lost in hard labels.

**Temperature $\tau$.** Higher temperature produces softer distributions (more uniform), amplifying the signal in the incorrect class probabilities. Too high: distributions become nearly uniform, losing all information. Common values: $\tau \in [2, 10]$ for language models; $\tau = 1$ reduces to standard cross-entropy.

---

## Intermediate

### Q4. What is movement pruning and why might it be more appropriate for fine-tuned models?

**Answer.**

Movement pruning (Sanh et al., 2020) prunes weights based on their gradient signal during fine-tuning, rather than their absolute value.

**Motivation.** Magnitude pruning retains large weights regardless of whether they are useful for the downstream task. In a pretrained model, some large weights may encode general language features that are not needed for a specific task (e.g., named entity recognition), while some small weights might be critical for the task.

**Movement pruning score.** For weight $w_{ij}$, the importance score is:

$$S_{ij} = w_{ij} \cdot \frac{\partial \mathcal{L}}{\partial w_{ij}}$$

This product is:
- Positive if the gradient suggests making the weight larger (the weight is "moving" toward a larger value and is thus important)
- Negative if the gradient suggests making the weight smaller (the weight is "moving" away from importance)

Weights with negative scores (those the gradient is pushing toward zero) are pruned.

**Smooth top-k with straight-through estimator.** Movement pruning uses soft masking during training:
$$\hat{w}_{ij} = w_{ij} \cdot m_{ij}, \quad m_{ij} = \text{Hard-Threshold}(S_{ij}, k)$$

The threshold is applied in the forward pass; the gradient flows through as if $m_{ij} = 1$ (straight-through estimator). This allows the scores $S_{ij}$ to be learned end-to-end.

**Comparison to magnitude pruning for fine-tuned BERT:**
- Magnitude pruning at $90\%$ sparsity: $\sim 4$ F1 drop on SQuAD
- Movement pruning at $90\%$ sparsity: $\sim 1.5$ F1 drop

The task-specific gradient signal makes movement pruning much more effective for fine-tuning scenarios.

---

### Q5. What are the key architectural choices when designing a student model for distillation?

**Answer.**

**Depth vs width reduction.** Two primary ways to shrink a model:

1. **Reduce depth** (fewer layers): e.g., distil a 12-layer model to 6 layers. Faster inference (linear in depth for sequential operations). Quality impact: removing layers reduces the model's ability to compute multi-step reasoning.

2. **Reduce width** (smaller hidden dim): e.g., reduce $d$ from 768 to 512. Reduces FFN and attention FLOPS (quadratically for FFN). Quality impact: reduces the model's representational capacity.

In practice, most distilled LLMs use depth reduction because:
- Decoder inference is inherently sequential (each layer must complete before the next), so depth directly determines latency per token
- Width reduction alone requires retraining from scratch; depth reduction can be initialised by skipping layers of the teacher

**Layer selection (DistilBERT approach).** When initialising the student with teacher layers, DistilBERT selects every other teacher layer. A student with $L/2$ layers is initialised from teacher layers $\{0, 2, 4, \ldots, L-2\}$, which provides a reasonable starting point.

**Intermediate layer distillation.** Beyond matching output distributions, the student can be trained to match teacher intermediate representations:

$$\mathcal{L}_{\text{hidden}} = \sum_{\ell} \|h_s^{(\ell)} - W_{\text{proj}} h_T^{(f(\ell))}\|_2^2$$

where $f(\ell)$ maps student layer $\ell$ to a teacher layer, and $W_{\text{proj}}$ projects to the teacher's hidden dimension if dimensions differ. This provides a richer training signal.

**Attention transfer.** Match attention patterns between teacher and student:

$$\mathcal{L}_{\text{attn}} = \sum_{\ell, h} \text{MSE}(A_s^{(\ell,h)}, A_T^{(f(\ell),h)})$$

Matching attention distributions helps the student learn the same information-routing patterns as the teacher.

---

### Q6. What is the distillation loss decomposition for language model distillation?

**Answer.**

For an autoregressive LM, the standard distillation loss operates at each token position:

$$\mathcal{L}_{\text{distill}} = -\sum_{t=1}^T \sum_{v=1}^V p_T(v \mid x_{<t}) \log p_S(v \mid x_{<t})$$

This is the cross-entropy of the student's distribution with respect to the teacher's distribution (a form of KL divergence: $\text{KL}(p_T \| p_S) = H(p_T, p_S) - H(p_T)$, and we can only optimise the cross-entropy term).

**Full loss in practice:**

$$\mathcal{L} = \alpha \mathcal{L}_{\text{LM}}(\text{student, hard labels}) + \beta \mathcal{L}_{\text{KD}}(\text{student, teacher logits}) + \gamma \mathcal{L}_{\text{hidden}}$$

- $\mathcal{L}_{\text{LM}}$: standard language modelling loss — ensures the student learns from the raw data
- $\mathcal{L}_{\text{KD}}$: KL divergence from teacher soft labels at temperature $\tau$ — transfers dark knowledge
- $\mathcal{L}_{\text{hidden}}$: intermediate representation matching — transfers structural knowledge

Typical values: $\alpha \in [0, 0.5]$, $\beta \in [0.5, 1.0]$, $\gamma \in [0, 0.5]$.

**Why $\mathcal{L}_{\text{LM}}$ is still useful alongside $\mathcal{L}_{\text{KD}}$.** The teacher's soft labels are only as good as the teacher. For rare tokens or inputs far from the teacher's training distribution, the teacher's logits may be noisy. The hard label loss provides a "ground truth" anchor, preventing the student from over-fitting to teacher errors.

---

## Advanced

### Q7. What is structured pruning of attention heads? How do you decide which heads to remove?

**Answer.**

Attention head pruning removes entire heads (one K, one Q, one V, and a slice of the output projection), producing a model with fewer heads that still runs with standard dense operations.

**Head importance scores.** Several criteria have been proposed:

**1. Gradient-based (Michel et al., 2019):** Multiply head outputs by a learned mask $m_h \in \{0, 1\}$ (or continuous $[0,1]$ during training). The importance is $|m_h \cdot \frac{\partial \mathcal{L}}{\partial m_h}|$ — how much quality loss removing head $h$ would cause.

**2. Attention entropy.** Heads with near-uniform attention (high entropy) attend to everything equally and likely carry little specialised information:
$$H_h = -\frac{1}{T}\sum_t \sum_j A_{h,t,j} \log A_{h,t,j}$$
Heads with the highest entropy are pruned first.

**3. Effect on output.** Directly ablate each head (set its output to 0 for all tokens in a validation set) and measure the quality drop. Heads causing the smallest drop are removed.

**Structured FFN neuron pruning.** Similar to head pruning, entire FFN neurons (rows in $W_{\text{up}}$, columns in $W_{\text{down}}$) can be pruned using activation statistics or gradient scores. The result is an FFN with reduced hidden dimension $d_{\text{ff}}' < d_{\text{ff}}$.

**Layer dropping (ShortGPT, LaCo).** The most aggressive structured pruning: remove entire transformer layers. LaCo (Large Language Model Compression) shows that for large models, layers with high cosine similarity between their input and output can be removed with minimal quality loss (the layer is approximately acting as an identity function).

---

### Q8. How do you evaluate the quality-efficiency trade-off when choosing between pruning, quantisation, and distillation?

**Answer.**

**Three dimensions of compression quality:**
1. **Accuracy/perplexity** on target tasks
2. **Inference latency** (time per token for decode)
3. **Memory footprint** (model + KV cache)

**Rough characterisation:**

| Method | Memory | Latency | Quality | Fine-tuning needed |
|---|---|---|---|---|
| INT4 quantisation (GPTQ/AWQ) | $4\times$ reduction | $2\times$ faster decode | $-1\text{--}3\%$ | No |
| INT8 quantisation (W8A8) | $2\times$ reduction | $1.5\text{--}2\times$ faster | $< 1\%$ | No |
| Structured pruning (50% heads) | $\sim 1.3\times$ reduction | $\sim 1.3\times$ faster | $-2\text{--}5\%$ | Yes (recovery) |
| Distillation ($2\times$ smaller) | $2\times$ reduction | $2\times$ faster | $-5\text{--}10\%$ | Yes (full train) |

**Choosing the right tool:**

- Need the same quality at half the memory, without retraining: **quantisation (GPTQ/AWQ INT4)** — strong quality retention, no fine-tuning
- Need the fastest possible latency for a target quality: **smaller distilled model** — a 7B distilled model is faster than a 13B quantised model at the same quality budget
- Fine-tuning a large model for a specific task with limited hardware: **QLoRA** — enables fine-tuning of $65\text{B}+$ models on a single GPU
- Deploying on hardware without quantisation support (edge devices): **structured pruning** — produces standard dense models

**Common production stack.** In practice, quantisation and distillation are often combined: distil a large model to a mid-size model, then quantise the distilled model. For example: teacher GPT-4 level (600B+) $\to$ student 7B via distillation $\to$ INT4 quantised 7B for deployment. Each step contributes multiplicatively to the compression ratio.
