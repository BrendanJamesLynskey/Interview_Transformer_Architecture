# Supervised Fine-Tuning (SFT): Interview Questions

Supervised Fine-Tuning is the first stage of the alignment pipeline. It trains a pretrained language model to follow instructions by learning from human-written demonstrations. SFT is foundational to every modern chat model and appears frequently in ML engineer and research interviews.

---

## Fundamentals

### Q1. What is supervised fine-tuning in the context of LLM alignment, and how does it differ from pretraining?

**Answer.**

Pretraining optimises a language model to predict the next token across a large, diverse corpus. The objective is purely distributional: the model learns to assign high probability to naturally occurring text. It has no concept of "following an instruction" — it is equally likely to continue a prompt with an unhelpful tangent as with a correct answer.

SFT shifts the model's behaviour toward a desired response distribution by continuing to train on a curated dataset of (prompt, response) pairs, where the responses are the behaviours you want the model to exhibit. The loss is still cross-entropy next-token prediction, but computed only over the response tokens (prompt tokens are masked from the loss).

| Dimension | Pretraining | SFT |
|---|---|---|
| Data scale | Trillions of tokens | Thousands to millions of examples |
| Data source | Web crawls, books, code | Human demonstrations, curated pairs |
| Loss mask | All tokens | Response tokens only |
| Learning rate | Scheduled from scratch | Small LR, few epochs |
| Objective | Model the distribution of text | Match the distribution of desired responses |

The model after SFT is sometimes called the "SFT model" or "policy initialisation" — it is the starting point for RLHF or DPO.

---

### Q2. What is instruction tuning, and why is it effective even with small datasets?

**Answer.**

Instruction tuning is a form of SFT where training examples are (instruction, output) pairs spanning a wide variety of task types: question answering, summarisation, translation, code generation, reasoning, etc. The key insight is that, because the pretrained model has already learned the underlying capabilities, it only needs to learn the *format* of responding to instructions — not to acquire new knowledge.

This explains why instruction tuning works with surprisingly small datasets. The FLAN paper (Wei et al., 2022) showed that tuning on ~60 task types enabled strong generalisation to unseen tasks via zero-shot prompting. InstructGPT used roughly 13,000 demonstration examples to dramatically improve GPT-3's instruction-following. The pretrained model already "knows" how to answer; SFT teaches it when and how to surface that knowledge.

A useful mental model: pretraining fills a knowledge warehouse; instruction tuning builds the front door and signage so users can find what they need.

---

### Q3. What is the chat template format and why does it matter?

**Answer.**

When training a chat model, multi-turn conversations must be serialised into a flat token sequence. Different models use different special tokens to delimit roles. For example, Llama-3 uses:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
4<|eot_id|>
```

The template matters for three reasons:

1. **Loss masking.** Only assistant turns should contribute to the training loss. If system/user tokens are included in the loss, the model learns to predict user messages — not the desired behaviour, and a waste of gradient signal.

2. **Generation stopping.** At inference time, generation must stop at the end-of-turn token. If the template is misapplied, the model will hallucinate additional turns.

3. **Template mismatch.** Fine-tuning with one template and serving with another causes sharp performance degradation. This is one of the most common production bugs when adapting open-weight models.

HuggingFace's `tokenizer.apply_chat_template()` handles serialisation, but you must understand the underlying format to debug template issues.

---

### Q4. What does it mean to mask the prompt tokens from the SFT loss, and what goes wrong if you do not?

**Answer.**

During SFT, the full (prompt + response) sequence is fed into the model and next-token predictions are made at every position. The training signal should come only from positions within the *response*, because those are the tokens we want the model to learn to generate.

If prompt tokens are included in the loss:

- The model receives gradient signal for predicting instruction/user-turn tokens, wasting capacity and potentially learning wrong things about how instructions look.
- In multi-turn conversations, all turns in the loss means the model learns to predict *user* messages, distorting the policy.
- For short responses with long prompts, the loss is dominated by prompt tokens and carries almost no alignment signal.

**Implementation:** create a `labels` tensor that is a copy of `input_ids`, then set all positions corresponding to prompt tokens to `-100`. PyTorch's `CrossEntropyLoss` ignores positions with label `-100` by default.

---

### Q5. What is catastrophic forgetting in the context of SFT, and what are the standard mitigations?

**Answer.**

Catastrophic forgetting is the phenomenon where fine-tuning on a narrow distribution causes the model to lose capabilities acquired during pretraining. In SFT this can manifest as:

- Reduced performance on benchmarks not represented in the SFT data (e.g., coding ability degraded by a conversation-heavy SFT set).
- Loss of factual recall because the SFT examples involve fewer diverse facts.
- Degraded few-shot learning ability.

Standard mitigations:

1. **Data mixing.** Include a fraction of pretraining data (or a representative sample) in the SFT training mix ("replay"). Typical ratios: 5–20% pretraining data by token count.

2. **Low learning rate.** Fine-tune at a much smaller LR than pretraining (often 10x–100x smaller), so each gradient step moves the weights only a small distance from the pretrained initialisation.

3. **Few epochs.** SFT datasets are small; running many epochs leads to overfitting on the narrow distribution. 1–3 epochs is standard.

4. **LoRA / parameter-efficient fine-tuning.** By only training low-rank adapter weights, the vast majority of the pretrained weights are frozen and cannot forget.

5. **Elastic Weight Consolidation (EWC).** Penalises changes to weights important during pretraining. Rarely used in practice due to computational cost of estimating the Fisher information matrix.

---

## Intermediate

### Q6. Explain LoRA. Why is it particularly well-suited to SFT?

**Answer.**

LoRA is a parameter-efficient fine-tuning method (Hu et al., 2021). The core observation is that weight updates during fine-tuning tend to have low intrinsic rank. Instead of updating the full weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA introduces two small trainable matrices:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

At initialisation, $A$ is random Gaussian and $B$ is zero, so $\Delta W = 0$ and the model starts from the exact pretrained weights. Only $A$ and $B$ are trained; $W$ is frozen.

Trainable parameter count drops from $d \times k$ to $r(d + k)$. For a typical attention projection with $d = k = 4096$ and $r = 16$, this is a reduction from 16.8M to 131K parameters per matrix.

Why LoRA suits SFT:

- **Compute.** Full fine-tuning a 70B model requires ~280GB GPU memory in BF16 before activations or optimiser states. LoRA makes this tractable on fewer GPUs.
- **Forgetting.** Frozen base weights prevent catastrophic forgetting by construction.
- **Mergeability.** LoRA adapters can be merged back: $W \leftarrow W + BA$, at inference time with zero latency overhead.
- **Multi-task.** Multiple LoRA adapters can be trained on different tasks and swapped at inference time without separate full model copies.

Common follow-up: "Where do you apply LoRA?" — Standard targets are the Q, K, V, and output projection matrices in attention, and the up/down projections in the MLP. Applying LoRA to all linear layers generally improves quality.

---

### Q7. How do you think about data quality vs data quantity in SFT?

**Answer.**

A small amount of high-quality data almost always outperforms a large amount of low-quality data in SFT. The LIMA paper (Zhou et al., 2023) demonstrated that 1,000 carefully curated examples could produce instruction-following quality competitive with models trained on orders of magnitude more data. The model already has the knowledge; alignment is a small directional nudge.

However, "quality" is multidimensional:

- **Diversity.** 1,000 examples of the same task type add less value than 1,000 examples spanning 50 task types. Coverage of the target use-case space matters more than sheer volume.
- **Response quality.** Responses should exemplify the exact behaviour you want: not too verbose, not sycophantic, factually correct, appropriately formatted.
- **Prompt difficulty.** Easy prompts the model already handles well contribute little gradient signal. Challenging prompts that expose behavioural gaps are more informative.
- **Consistency.** Contradictory instructions (some examples reward conciseness, others reward verbosity) produce a muddled policy.

In practice: start with a small, carefully curated seed dataset. Measure model performance. Add data where performance is weakest. Avoid scraping large quantities of unfiltered synthetic or web data.

Useful heuristic: if you cannot describe what makes a "good" response in your domain, you are not ready to collect data yet. Quality requires a clear rubric.

---

### Q8. What is a data mixture strategy for SFT, and how do you decide the proportions?

**Answer.**

A data mixture combines multiple source datasets, each with a sampling weight in the final SFT corpus. Sources may include:

- Human-written demonstrations (high quality, expensive, limited scale)
- Synthetic data from a stronger teacher model
- Domain-specific data (medical, legal, coding)
- Safety and refusal data
- Pretraining replay data

A common sampling scheme uses temperature scaling:

$$p_i \propto n_i^{1/T}$$

where $n_i$ is the raw size of source $i$ and $T > 1$ upsamples smaller sources relative to larger ones. $T = 1$ gives proportional sampling; $T \to \infty$ gives uniform sampling across sources.

Decision process:

1. **Define capabilities.** List the target capabilities (math, coding, conversation, safety, etc.).
2. **Evaluate baselines.** Run evals on each capability with the base model.
3. **Allocate proportions.** Give higher weight to capabilities with the largest gap between baseline and target.
4. **Ablate.** Train on small-scale mixture variants and measure capability tradeoffs. Most capabilities compete for weight updates.
5. **Safety floor.** Always maintain a minimum proportion of safety/refusal data. Underweighting safety data is difficult to recover from.

Common mistake: too much synthetic data from a single teacher model. The student can "learn the teacher's voice," exhibiting teacher-specific artefacts (e.g., GPT-4's characteristic hedging phrases).

---

### Q9. What evaluation metrics do you use to assess an SFT model, and what are their limitations?

**Answer.**

**Automatic metrics:**

- **Benchmark accuracy** (MMLU, HumanEval, GSM8K): useful for measuring capability retention and specific skill improvements, but benchmarks can be saturated or contaminated.
- **MT-Bench / AlpacaEval:** automatic evaluation using a judge LLM (e.g., GPT-4) to score responses on helpfulness. Faster than human eval but inherits the judge's biases — length preference, self-preference, positional bias.
- **Perplexity on held-out data:** measures fit to the training distribution but is a poor proxy for instruction-following quality.

**Human evaluation:**

- **Head-to-head comparisons:** ask annotators which of two responses they prefer. More reliable than absolute scores, but expensive.
- **Rubric-based scoring:** structured criteria (helpfulness, correctness, harmlessness, conciseness) rated on a scale. Requires annotator training and has high inter-annotator variance.

**Limitations:**

- Benchmarks can be overfit without genuine capability improvement.
- Judge-LLM evaluation has systematic biases: longer responses and responses stylistically similar to the judge model score higher.
- None of these metrics capture real-world deployment quality, which requires A/B testing.

Best practice: maintain a diverse eval suite (automated benchmarks + LLM judge + periodic human eval) and track trends across training iterations rather than relying on a single number.

---

## Advanced

### Q10. Explain the gradient dynamics of SFT. Why do you typically train for only 1–3 epochs?

**Answer.**

The SFT dataset is a tiny distribution compared to pretraining. When you fine-tune for many epochs, you are repeatedly optimising on the same small set of examples. Several pathological things happen:

1. **Overfitting to response format.** The model memorises specific phrasings from the training set and applies them rigidly even when inappropriate.
2. **Loss of calibration.** Log-probability on training responses becomes near-perfect, but the model's uncertainty estimates on out-of-distribution inputs degrade.
3. **Distributional shift from pretraining.** With many gradient steps, the KL divergence from the pretrained model grows, moving the model into a narrow valley specific to the SFT data. This manifests as repetition, incoherence on complex queries, or overconfidence.

The optimal stopping point is where the SFT loss on a held-out validation set (covering diverse tasks, not just the training distribution) flattens or begins to rise. In practice this is almost always within 1–3 epochs.

**Loss landscape interpretation:** the pretrained model sits in a wide, flat minimum. SFT exerts a directional force toward the SFT data distribution. After a few gradient steps, the model is still in the neighbourhood of the pretrained minimum (good generalisation). After many steps, it has moved into a narrow valley specific to the SFT data, with poor off-manifold generalisation.

---

### Q11. How does data contamination affect SFT evaluations, and how would you detect and mitigate it?

**Answer.**

Data contamination occurs when examples from evaluation benchmarks appear verbatim or near-verbatim in the training data. SFT datasets assembled from web scrapes or large synthetic pipelines are especially vulnerable because popular benchmark problems are widely reproduced online.

**Detection methods:**

1. **N-gram overlap.** Compute the fraction of n-grams (typically 13-gram) in each benchmark example that appear in the training corpus. High overlap flags potential contamination (Brown et al., 2020, used this for GPT-3).
2. **String matching.** Exact-match search of benchmark answers in training data. No false positives, but lower recall than n-gram overlap.
3. **Membership inference.** Train two models — one with and one without suspected contaminated data — and compare their log-likelihoods on the benchmark. Significantly lower perplexity on the contaminated model confirms the data was seen during training.
4. **Canary injection.** At training time, inject known fake "benchmark" examples with deliberately wrong answers. If the model produces wrong answers on canaries at eval time, the canary was memorised, confirming that memorisation of benchmark content is occurring.

**Mitigation:**

- Deduplicate training data against eval benchmarks before SFT.
- Use private held-out evals not released publicly.
- Report contamination statistics alongside benchmark results.
- Weight benchmark results by contamination rate when drawing conclusions.

---

### Q12. Describe the alignment tax. When does SFT hurt capabilities and how do you minimise the tradeoff?

**Answer.**

The alignment tax refers to the empirical observation that SFT (and subsequent RLHF/DPO) sometimes degrades performance on capability benchmarks even while improving instruction-following and safety. The tax is most pronounced when:

1. The SFT dataset is heavily skewed toward conversational or short-form responses, shifting the model away from the long-form, reasoning-heavy distribution of capability benchmarks.
2. The learning rate is too large, moving the model far from the pretrained weights.
3. The SFT dataset contains factual errors or low-quality responses that overwrite correct pretrained knowledge.

**Minimising the tradeoff:**

- **Capability-preserving data.** Include SFT examples that exercise the model's hardest capabilities (math proofs, complex code, multi-step reasoning). These simultaneously improve instruction-following and reinforce capability.
- **Pretraining replay.** Mixing pretraining data into SFT keeps the model anchored to the pretraining distribution.
- **Low LR + early stopping.** The alignment tax is minimised near the early stopping point.
- **LoRA.** Freezing the base weights by construction avoids capability degradation.
- **Evaluation-aware data collection.** Continuously measure capability benchmarks during SFT dataset curation. If adding a new data source causes a benchmark drop, investigate before proceeding.

The LIMA paper's finding — 1,000 examples, competitive quality — suggests the alignment tax is mostly a data quality problem: a small, diverse, high-quality dataset extracts alignment without sacrificing capabilities.

---

### Q13. How would you build a scalable, high-quality SFT data pipeline from scratch?

**Answer.**

A robust pipeline has several stages:

**1. Requirement definition**
Define target capabilities and a corresponding evaluation suite before collecting any data. Write a detailed annotation rubric: what constitutes an ideal response for each capability category?

**2. Prompt sourcing**
- *Seed prompts:* collect real user queries from internal logs or public sources (ShareGPT, WildChat). High ecological validity but require filtering for quality, safety, and PII.
- *Synthetic prompts:* use a strong model to generate diverse prompts from a topic taxonomy. Control diversity by seeding from a capability tree.
- *Adversarial prompts:* generate prompts that specifically target known weaknesses (jailbreaks, edge cases, low-resource languages).

**3. Response generation and selection**
- For expensive capabilities: hire domain experts to write responses (math, medical, legal).
- For scale: generate multiple responses per prompt from a strong teacher model, then filter by quality using a classifier or LLM judge.

**4. Deduplication and balance**
- Deduplicate prompts (near-dedup with MinHash) to avoid the model memorising specific phrasings.
- Balance across capability categories to avoid over-representation.
- Check and remove benchmark contamination.

**5. Quality audits**
- Random-sample review by humans at each stage.
- Track inter-annotator agreement. Low agreement means the rubric is ambiguous — fix the rubric before continuing.

**6. Versioning and lineage**
Track which data version produced which model. Data bugs are the most common cause of unexplained performance regressions.

**7. Continuous improvement loop**
Deploy the SFT model, collect user feedback on its failures, use failure cases to generate new training data. This is the data flywheel.
