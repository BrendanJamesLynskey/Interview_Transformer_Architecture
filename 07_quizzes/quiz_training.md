# Quiz: Pretraining, Tokenisation, and Training Methodology

**18 multiple-choice questions** covering causal language modelling vs masked language modelling,
byte-pair encoding, learning rate schedules, Chinchilla scaling laws, mixed-precision training,
and gradient clipping.

Difficulty: Fundamentals through Advanced.

---

## Questions

---

### Q1 — Causal language modelling (CLM) objective

Which loss function is used in causal language modelling?

**A.** Mean squared error between predicted and target token embeddings.

**B.** Cross-entropy loss predicting the next token given all preceding tokens:
$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$.

**C.** Cross-entropy loss predicting masked tokens given their surrounding context.

**D.** Contrastive loss between positive and negative sequence pairs.

---

### Q2 — Masked language modelling (MLM) objective

BERT uses masked language modelling.  In the original BERT paper, what fraction of tokens are
masked per sequence, and how are those positions handled?

**A.** 50% of tokens are replaced with `[MASK]` and the model must predict all of them.

**B.** 15% of tokens are selected; of those, 80% are replaced with `[MASK]`, 10% with a random
token, and 10% are left unchanged.  The model predicts only the selected positions.

**C.** 15% of tokens are replaced with `[MASK]` and the model predicts every token in the
sequence, including unmasked ones.

**D.** 20% of tokens are masked; the model predicts only even-indexed positions.

---

### Q3 — CLM vs MLM: downstream task suitability

Which statement best characterises when to prefer CLM pretraining over MLM pretraining?

**A.** CLM is preferred for classification tasks because it produces a single vector
representation per sequence.

**B.** CLM is preferred for generative tasks (e.g., open-ended text generation, summarisation,
code completion) because it trains the model to produce coherent continuations; MLM is preferred
for tasks requiring rich bidirectional context (e.g., NER, sentence classification).

**C.** MLM should always be preferred because it trains on more signal per batch (only ~15% of
tokens are used for loss in CLM).

**D.** CLM and MLM produce identical representations; the choice is purely a matter of convention.

---

### Q4 — Byte-pair encoding (BPE)

Which description correctly summarises the BPE tokenisation algorithm?

**A.** Split each word into characters; then iteratively merge the most frequent bigram of
consecutive symbols until the vocabulary reaches the target size.

**B.** Train a unigram language model; iteratively remove tokens with the lowest marginal
likelihood contribution.

**C.** Split text into substrings of fixed length $n$ characters.

**D.** Use WordPiece: greedily split unknown words into the longest known vocabulary substrings.

---

### Q5 — BPE vocabulary size trade-off

When choosing BPE vocabulary size, which pair of statements is correct?

**A.** Smaller vocabulary -> fewer tokens per sequence, faster inference; but each token
carries less information, harming rare-word coverage.

**B.** Larger vocabulary -> fewer tokens per sequence (strings are represented by fewer, longer
tokens) but larger embedding tables and output projection layers.

**C.** Vocabulary size has no effect on model parameter count.

**D.** Smaller vocabulary -> the embedding table is larger because each token is longer.

---

### Q6 — Learning rate warm-up

Most large Transformer training runs use a linear learning rate warm-up phase.  Why?

**A.** Warm-up allows the model to overfit briefly to the first batches, providing a better
initialisation.

**B.** At initialisation, weight gradients are unreliable (large variance, poorly scaled)
because the model has not yet seen enough data.  Starting with a small LR and ramping up
prevents large early updates from pushing parameters into poor regions.

**C.** Warm-up is required by the Adam optimiser to initialise its first and second moment
estimates before they are reliable.

**D.** Warm-up reduces the total number of gradient steps needed, speeding up convergence.

---

### Q7 — Cosine learning rate decay

After warm-up, many LLM training runs use cosine decay.  What does the learning rate look like
at the midpoint of decay?

**A.** $\eta_{\min}$, the minimum learning rate.

**B.** $\eta_{\max}$, unchanged from the peak.

**C.** Approximately $\frac{\eta_{\max} + \eta_{\min}}{2}$, the midpoint between maximum and
minimum LR.

**D.** $\frac{\eta_{\max}}{\sqrt{T_{\text{decay}}}}$, following the inverse square root schedule.

---

### Q8 — Chinchilla scaling law (compute-optimal training)

The Chinchilla paper (Hoffmann et al., 2022) revised earlier scaling advice.  What is its key
finding about the compute-optimal number of training tokens $D$ for a model with $N$ parameters?

**A.** $D \approx 10N$ -- models are trained for roughly 10 tokens per parameter.

**B.** $D \approx 20N$ -- for a compute-optimal run, models should be trained on roughly 20
tokens per parameter, balancing model size and data.

**C.** $D \approx N^2$ -- larger models need quadratically more data.

**D.** $D$ is independent of $N$; the optimal data size is determined only by hardware budget.

---

### Q9 — Chinchilla implication for GPT-3

GPT-3 has approximately 175 billion parameters and was trained on roughly 300 billion tokens.
What does Chinchilla suggest about GPT-3's training?

**A.** GPT-3 was overtrained -- too many tokens for its model size.

**B.** GPT-3 was undertrained -- given its compute budget, a smaller model trained on more
tokens would have achieved better performance.

**C.** GPT-3 followed the Chinchilla-optimal recipe exactly.

**D.** GPT-3's training is irrelevant to Chinchilla because the laws only apply to models below
10B parameters.

---

### Q10 — Mixed-precision training (FP16 / BF16)

Which statement about mixed-precision training is accurate?

**A.** Weights are stored entirely in FP16, cutting memory in half with no accuracy cost.

**B.** A master copy of weights is kept in FP32; gradients and activations are computed in FP16
(or BF16).  A loss scaling factor multiplies the loss before the backward pass to prevent FP16
underflow in small gradients.

**C.** Mixed precision trains on BF16 activations but uses INT8 for gradient accumulation.

**D.** FP16 has a wider dynamic range than FP32, which is why it is preferred for gradient
computation.

---

### Q11 — BF16 vs FP16

BF16 (Brain Float 16) is preferred over FP16 in many modern LLM training runs.  Why?

**A.** BF16 provides higher precision (more mantissa bits) than FP16.

**B.** BF16 has the same 8-bit exponent as FP32, giving the same dynamic range and avoiding the
overflow/underflow issues that require loss scaling in FP16 training.

**C.** BF16 computations are natively faster on all hardware.

**D.** BF16 requires less memory than FP16 because it uses a 14-bit representation.

---

### Q12 — Gradient clipping

Gradient clipping by global norm is a standard technique in LLM training.  What does it do?

**A.** It sets any gradient component larger than a threshold $\tau$ to exactly $\tau$.

**B.** It computes the global $\ell_2$ norm of all gradients; if it exceeds threshold $\tau$,
all gradients are scaled down proportionally so the global norm equals $\tau$.  This preserves
the gradient direction.

**C.** It removes gradient updates for parameters whose gradients exceed the threshold,
effectively freezing those parameters for that step.

**D.** It clips the loss value before the backward pass so large losses do not propagate.

---

### Q13 — Gradient accumulation

A researcher wants an effective batch size of 1024 but can only fit 64 samples per GPU step.
How does gradient accumulation solve this?

**A.** Run 16 forward-backward passes without calling the optimiser update; sum the gradients
across those 16 mini-batches; then call the optimiser once.  The effective gradient equals that
of a single batch of 1024 samples.

**B.** Duplicate each mini-batch 16 times to simulate a larger batch.

**C.** Reduce the learning rate by a factor of 16 and run the optimiser every step.

**D.** Gradient accumulation is only possible with data parallelism across 16 GPUs.

---

### Q14 — Adam optimiser in Transformer training

Why is Adam (or AdamW) the standard optimiser for Transformer pretraining rather than SGD with
momentum?

**A.** Adam converges in fewer epochs than SGD for convex objectives.

**B.** Adam maintains per-parameter adaptive learning rates derived from first and second moment
estimates of gradients, which handles the very different gradient scales across embedding
parameters, attention weights, and FFN weights common in Transformers.

**C.** Adam does not require a learning rate schedule; it automatically decays the LR.

**D.** SGD is incompatible with FP16 precision, whereas Adam is not.

---

### Q15 — AdamW weight decay

AdamW decouples weight decay from the gradient update.  What problem does this solve compared
to L2 regularisation applied directly to the loss in vanilla Adam?

**A.** L2 regularisation increases memory consumption; AdamW removes this overhead.

**B.** In Adam, adding L2 to the loss modifies the gradient, which is then divided by the
adaptive second moment estimate, effectively reducing weight decay for parameters with large
gradient variance.  AdamW applies weight decay directly to the weights, independent of the
gradient scale.

**C.** AdamW doubles the effective weight decay to improve generalisation.

**D.** Standard Adam with L2 regularisation diverges on all Transformer architectures.

---

### Q16 — Perplexity as a training metric

Perplexity (PPL) is commonly reported for language model evaluation.  What does PPL measure?

**A.** The fraction of tokens predicted correctly (token accuracy).

**B.** The exponentiated average negative log-likelihood per token:
$\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t \mid x_{<t})\right)$.
Lower is better; a perplexity of $k$ means the model is as uncertain as choosing uniformly
among $k$ options at each step.

**C.** The geometric mean of sentence-level probabilities across the test set.

**D.** The average number of tokens the model generates before making its first error.

---

### Q17 — Data deduplication

Why is exact or near-duplicate removal from pretraining corpora important?

**A.** Duplicate documents increase disk storage requirements but do not affect model quality.

**B.** Models trained on deduplicated data exhibit lower test loss on held-out data, better
generalisation, and less memorisation of training examples; deduplication also prevents
evaluation data leakage when test sets share content with the web corpus.

**C.** Deduplication is only important for instruction-tuning datasets, not pretraining corpora.

**D.** Duplicate removal is detrimental because it reduces dataset diversity.

---

### Q18 — Training loss spike recovery

During a long pretraining run a sudden spike in loss occurs.  What is the most common
practical response?

**A.** Immediately stop training and restart from scratch with a lower learning rate.

**B.** Increase the learning rate to escape the local minimum that caused the spike.

**C.** Roll back to a recent checkpoint before the spike, inspect the data batch that caused it
(often a corrupted or out-of-distribution shard), remove or repair the problematic data, and
resume training from the checkpoint.

**D.** Loss spikes are expected and always self-correct within a few steps; no action is needed.

---

## Answer Key

| Q  | Answer |
|----|--------|
| 1  | B      |
| 2  | B      |
| 3  | B      |
| 4  | A      |
| 5  | B      |
| 6  | C      |
| 7  | C      |
| 8  | B      |
| 9  | B      |
| 10 | B      |
| 11 | B      |
| 12 | B      |
| 13 | A      |
| 14 | B      |
| 15 | B      |
| 16 | B      |
| 17 | B      |
| 18 | C      |

---

## Detailed Explanations

---

### Q1 — Causal language modelling objective

**Correct: B.**

CLM maximises the likelihood of the training sequence under the autoregressive factorisation:
$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$.  The training loss is the average
negative log-likelihood (cross-entropy) over all token positions.  This is the objective used
by GPT-1/2/3, LLaMA, Falcon, Mistral, and virtually all modern open-weights LLMs.

- **A is wrong.** MSE over embeddings is not a standard LM objective; it would require a
  meaningful distance metric in embedding space, which is not guaranteed.
- **C is wrong.** That describes MLM (BERT-style), not CLM.
- **D is wrong.** Contrastive objectives are used for representation learning (e.g., SimCSE,
  CLIP) but not for standard autoregressive language modelling.

---

### Q2 — MLM masking procedure

**Correct: B.**

The 80/10/10 split was motivated in the BERT paper: always masking with `[MASK]` would create a
train-test mismatch (the `[MASK]` token never appears at inference time).  The 10% random
replacement forces the model to rely on context rather than just looking for `[MASK]` markers.
The 10% unchanged helps the model learn good representations for all tokens, not just masked ones.

- **A is wrong.** 50% masking would destroy too much context.
- **C is wrong.** BERT only computes loss at the selected (masked) positions.
- **D is wrong.** The fraction is 15%, not 20%, and the positional restriction is incorrect.

---

### Q3 — CLM vs MLM downstream task suitability

**Correct: B.**

CLM is naturally suited to generation: the model is trained to extend a prefix.  MLM produces
richer bidirectional token representations because every token can attend to the full context,
making it advantageous for discriminative tasks where no generation is required.

- **A is wrong.** Producing a single classification vector is done by pooling encoder hidden
  states; CLM models can be adapted but are not inherently superior for classification.
- **C is wrong.** This reverses the signal-per-batch comparison.  In CLM, EVERY token position
  contributes to the loss; in MLM, only ~15% of positions do.  CLM actually uses more loss signal
  per batch.
- **D is wrong.** The representations and downstream task suitability differ substantially
  (bidirectional vs unidirectional context).

---

### Q4 — Byte-pair encoding

**Correct: A.**

BPE was introduced for NMT by Sennrich et al. (2016), adapted from a data compression algorithm.
Starting from a character vocabulary, the algorithm greedily merges the most frequent symbol
pair until the target vocabulary size is reached.

- **B is wrong.** That describes the SentencePiece unigram language model algorithm.
- **C is wrong.** Fixed-length character $n$-grams is a simpler, less effective method not used
  in practice for LLMs.
- **D is wrong.** WordPiece (used by BERT) is a related but distinct algorithm that maximises
  language model likelihood rather than merge frequency.

---

### Q5 — BPE vocabulary size trade-off

**Correct: B.**

Larger vocabularies -> fewer tokens represent the same text -> shorter sequences -> faster
attention (which is $O(T^2)$) -> faster inference.  The cost is a larger embedding matrix
($V \times d$) and a larger output projection ($d \times V$), both of which increase memory
and the parameter count.

- **A is wrong.** It has the direction backwards: SMALLER vocabulary -> MORE tokens per sequence
  (longer sequences), not fewer.
- **C is wrong.** Vocabulary size directly affects the embedding table and LM head size; for a
  model like GPT-2 with $V = 50{,}257$ and $d = 1024$, the embedding table alone is ~200M
  parameters.
- **D is wrong.** Smaller vocabulary means shorter average token length and MORE tokens per
  sentence, not a larger table.

---

### Q6 — Learning rate warm-up

**Correct: C.**

At the start of training, Adam's moment estimates ($m_t$ and $v_t$) are initialised to zero.
For the first few steps these estimates are biased toward zero, causing Adam to take
inappropriately scaled steps.  Warm-up allows the estimates to stabilise before large gradient
steps are taken.  Additionally, gradients are noisiest early in training, so small initial
updates reduce the chance of catastrophic early loss increases.

- **A is wrong.** Deliberate early overfitting is not a sound training strategy; warm-up is
  not intended to encourage overfitting.
- **B is wrong.** While Adam bias correction is a related concern, the primary motivation given
  in the original Transformer paper and subsequent LLM training guides is gradient noise at
  initialisation.
- **D is wrong.** Warm-up does not reduce the total number of gradient steps; it only shapes
  the learning rate schedule.

---

### Q7 — Cosine decay midpoint

**Correct: C.**

The cosine decay formula is:

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

At $t = T/2$, $\cos(\pi/2) = 0$, so $\eta(T/2) = \eta_{\min} + \frac{1}{2}(\eta_{\max} -
\eta_{\min}) = \frac{\eta_{\max} + \eta_{\min}}{2}$.

- **A is wrong.** $\eta_{\min}$ is reached at the END of decay ($t = T$).
- **B is wrong.** $\eta_{\max}$ is the starting value; the LR decreases monotonically.
- **D is wrong.** That formula describes an inverse-square-root (Noam) schedule, not cosine decay.

---

### Q8 — Chinchilla scaling law

**Correct: B.**

Chinchilla found $N \propto C^{0.5}$ and $D \propto C^{0.5}$ where $C$ is compute (FLOPs),
implying $D \approx 20N$ for optimal allocation.  This overturned the Kaplan et al. (2020)
finding that suggested scaling model size more aggressively was more important than data.

- **A is wrong.** 10 tokens/parameter is well below the Chinchilla optimum; the model would
  be undertrained.
- **C is wrong.** The relationship is linear ($D \approx 20N$), not quadratic.
- **D is wrong.** Both $N$ and $D$ are jointly determined by compute; they are not independent.

---

### Q9 — Chinchilla and GPT-3

**Correct: B.**

With $N = 175 \times 10^9$, Chinchilla-optimal training would use approximately $20 \times 175
\times 10^9 = 3.5 \times 10^{12}$ tokens.  GPT-3 used only $3 \times 10^{11}$ tokens -- roughly
10x less than the Chinchilla optimum.  Given the same compute, a model of roughly $\sim$70B
parameters trained on $\sim$1.4T tokens would have outperformed GPT-3 (this is essentially the
Chinchilla model itself).

- **A is wrong.** "Overtrained" would mean too many tokens, which is the opposite of GPT-3's
  situation.
- **C is wrong.** GPT-3 predates the Chinchilla analysis and used significantly fewer tokens
  than the optimum.
- **D is wrong.** Chinchilla's laws were derived from runs ranging across many orders of
  magnitude in compute and model size, explicitly including the GPT-3 scale.

---

### Q10 — Mixed-precision training

**Correct: B.**

The standard mixed-precision recipe (Micikevicius et al., 2018) maintains FP32 master weights,
computes the forward and backward pass in FP16, and applies a dynamic loss scaling factor to
prevent gradients from underflowing to zero in FP16 (which has a minimum representable value of
$\sim 6 \times 10^{-8}$, much larger than FP32's $\sim 10^{-38}$).

- **A is wrong.** Storing weights only in FP16 without an FP32 master copy typically leads to
  accumulated rounding errors that hurt final model quality.
- **C is wrong.** INT8 gradients would be extremely lossy; INT8 is used for inference
  quantisation, not training.
- **D is wrong.** FP16 has a NARROWER dynamic range than FP32 (5 exponent bits vs. 8), which is
  precisely why loss scaling is necessary.

---

### Q11 — BF16 vs FP16

**Correct: B.**

BF16 uses 8 exponent bits (identical to FP32) and 7 mantissa bits, versus FP16's 5 exponent
bits and 10 mantissa bits.  The wider exponent gives BF16 the same dynamic range as FP32
($\sim 3.4 \times 10^{38}$), eliminating the overflow/underflow problem that necessitates loss
scaling in FP16 training.  The cost is lower precision (7 mantissa bits vs 10), but this is
generally acceptable for gradient computations.

- **A is wrong.** BF16 has FEWER mantissa bits (7 vs 10), so lower precision than FP16.
- **C is wrong.** BF16 has hardware support on modern Ampere/Hopper GPUs, but it is not
  universally faster on all hardware.
- **D is wrong.** Both FP16 and BF16 are 16-bit formats, using the same memory per element.

---

### Q12 — Gradient clipping

**Correct: B.**

Global norm clipping scales all gradients by $\min(1, \tau / \|g\|_2)$ where $\|g\|_2$ is the
global gradient norm across all parameters.  This preserves the gradient direction while
bounding its magnitude.  It is distinct from component-wise clipping (option A), which changes
the gradient direction.

- **A is wrong.** Component-wise clipping changes gradient direction and can cause pathological
  behaviour (e.g., certain dimensions update more than others for no principled reason).
- **C is wrong.** Freezing parameters with large gradients would prevent the model from updating
  the weights that need the most correction.
- **D is wrong.** Clipping the loss itself before the backward pass is not standard practice
  and would distort the loss landscape.

---

### Q13 — Gradient accumulation

**Correct: A.**

Gradient accumulation sums $\nabla_\theta \mathcal{L}$ over $k$ mini-batches before calling
`optimizer.step()`.  Because the loss is averaged (or summed) over tokens, the accumulated
gradient is equivalent to computing the gradient over all $k$ mini-batches simultaneously.
This is mathematically identical to a larger batch size without the memory requirement.

- **B is wrong.** Duplicating batches introduces redundant data and biases the gradient estimate.
- **C is wrong.** Reducing the LR changes the effective update magnitude but does not produce
  the same gradient signal as a larger batch.
- **D is wrong.** Gradient accumulation works on a single device; multi-GPU data parallelism is
  a separate (and complementary) technique.

---

### Q14 — Adam vs SGD

**Correct: B.**

Transformers have heterogeneous parameter groups with vastly different gradient scales: embedding
rows corresponding to rare tokens receive sparse, small gradients, while frequently activated FFN
weights may receive dense, larger gradients.  Adam's adaptive per-parameter step sizes naturally
accommodate this diversity.  SGD with a single global LR struggles to balance these scales without
extensive hyperparameter tuning.

- **A is wrong.** This reverses the comparison; SGD with momentum can match Adam on convex
  problems, but Transformer loss landscapes are not convex.
- **C is wrong.** Adam still requires an LR schedule; its adaptive scaling changes the effective
  per-parameter LR but does not eliminate the need for a global LR.
- **D is wrong.** Both Adam and SGD can be implemented in FP16 precision.

---

### Q15 — AdamW weight decay

**Correct: B.**

In standard L2 regularisation, the gradient of the regularisation term $\lambda \|w\|^2$ adds
$2\lambda w$ to the raw gradient.  Adam then divides this combined gradient by $\sqrt{\hat{v}}$
(the adaptive second moment), effectively scaling down weight decay for high-variance parameters.
AdamW applies the weight decay update multiplicatively and separately: $w \leftarrow (1 -
\lambda \eta) w$ after the Adam gradient step, restoring the intended regularisation strength.

- **A is wrong.** L2 regularisation is computed during the backward pass and adds minimal memory
  overhead.
- **C is wrong.** AdamW does not double the weight decay; it restores the intended magnitude.
- **D is wrong.** Standard Adam with L2 does not diverge; it simply applies suboptimal weight
  decay scaling.

---

### Q16 — Perplexity

**Correct: B.**

Perplexity is the exponentiated cross-entropy: $\text{PPL} = e^{\mathcal{L}}$.  A model with
PPL = 10 is as uncertain as a uniform distribution over 10 options at each step.  For reference,
a character-level model on English text might achieve PPL ~2--4 bits/char; GPT-3 on Penn
Treebank achieves PPL ~20.

- **A is wrong.** Token accuracy tracks exact top-1 predictions; perplexity accounts for
  the full probability distribution.
- **C is wrong.** Sentence-level geometric mean would not match the standard per-token PPL
  definition.
- **D is wrong.** Perplexity is a probabilistic measure, not an error-count measure.

---

### Q17 — Data deduplication

**Correct: B.**

Lee et al. (2022) "Deduplicating Training Data Makes Language Models Better" showed that
deduplication at multiple granularities (exact, near-duplicate, document-level) consistently
improves held-out perplexity.  Memorisation of training data also poses privacy and copyright
risks.  Leakage of benchmark data (e.g., test sets scraped from the web) can inflate reported
evaluation scores.

- **A is wrong.** Duplicates actively harm model quality by causing the model to memorise
  specific examples and by biasing the data distribution.
- **C is wrong.** Deduplication of pretraining corpora (which can be terabytes in size) has
  been shown to matter at scale, sometimes more than for smaller fine-tuning datasets.
- **D is wrong.** Diversity is measured by information content, not token count; duplicate
  documents add no new information and distort the effective data distribution.

---

### Q18 — Training loss spike recovery

**Correct: C.**

Loss spikes in large-scale training are often caused by a single bad data batch (corrupted
text, encoding errors, extreme-length sequences, or near-zero loss gradients from adversarial
content).  Rolling back to a recent checkpoint, identifying and removing the problematic shard,
then resuming is the standard mitigation practised at major labs (documented in training reports
for PaLM, LLaMA, etc.).

- **A is wrong.** Restarting from scratch is extremely expensive and usually unnecessary; the
  checkpoint is a cheaper recovery point.
- **B is wrong.** Increasing the LR during a spike would likely cause divergence.
- **D is wrong.** Some spikes do self-correct, but large ones (e.g., loss doubling) typically
  do not and indicate a real data or numerical issue that must be addressed.
