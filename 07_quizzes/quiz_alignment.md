# Quiz: Alignment — SFT, RLHF, DPO, and Constitutional AI

**18 multiple-choice questions** covering supervised fine-tuning, reward model training, PPO,
KL divergence penalty, Direct Preference Optimisation, reward hacking, and RLAIF.

Difficulty: Intermediate through Advanced.

---

## Questions

---

### Q1 — Purpose of alignment fine-tuning

After pretraining, why is an additional alignment phase (SFT + RLHF or DPO) necessary?

**A.** Pretraining is unsupervised, so the model cannot produce text at all without fine-tuning.

**B.** Pretraining optimises next-token prediction on web data, which teaches the model to
imitate the average distribution of internet text, including harmful, biased, or unhelpful
content.  Alignment fine-tuning teaches the model to follow instructions, be helpful, and avoid
harmful outputs -- objectives not directly optimised during pretraining.

**C.** Alignment is required to reduce the model's parameter count for efficient deployment.

**D.** Pretraining produces a model that outputs only one token at a time; alignment enables
multi-token generation.

---

### Q2 — Supervised fine-tuning (SFT) data format

In SFT for instruction following, what is the typical data format?

**A.** (input token, output token) pairs for next-token prediction, identical to pretraining.

**B.** (instruction, desired response) pairs -- sometimes structured as a conversation with
system prompt, user turn, and assistant turn.  The loss is computed on the assistant's response
tokens only, not on the instruction tokens.

**C.** (document, summary) pairs where the model must compress the input to a fixed length.

**D.** Pairs of (harmful prompt, empty response) that train the model to refuse all inputs.

---

### Q3 — Reward model training

In the RLHF pipeline (Ouyang et al., 2022, InstructGPT), how is the reward model trained?

**A.** The reward model is pretrained from scratch on a large text corpus to predict human
ratings from 1 to 10.

**B.** A pretrained LM (typically the SFT model) is fine-tuned as a regression head on a
dataset of human preference comparisons.  For each prompt, annotators rank two or more model
responses; the reward model is trained to assign higher scalar reward to the preferred response
using a pairwise loss such as the Bradley-Terry model.

**C.** The reward model is trained by the RL policy itself through self-play, using win/loss
signals from a rule-based judge.

**D.** The reward model is an ensemble of $n$ classifiers; its output is the mean of binary
"good / bad" classifications.

---

### Q4 — Bradley-Terry reward model loss

The reward model in RLHF is commonly trained with the following pairwise loss:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

where $y_w$ is the preferred (winning) response and $y_l$ the dispreferred (losing) response.
What property does this loss enforce?

**A.** It forces $r_\theta(x, y_w) = 1$ and $r_\theta(x, y_l) = 0$ for all pairs.

**B.** It encourages the reward model to assign a higher scalar score to the preferred response
than to the dispreferred one.  The sigmoid of the score difference models the probability that
$y_w$ is preferred, and the cross-entropy loss maximises that probability.

**C.** It minimises the mean squared error between the reward and a human rating score.

**D.** It ensures the reward is always in the range $[0, 1]$.

---

### Q5 — PPO in RLHF

RLHF uses Proximal Policy Optimisation (PPO) to fine-tune the language model.  What is the
policy, action, state, and reward in this formulation?

**A.** Policy = reward model; action = token; state = prompt; reward = reward model score.

**B.** Policy = the LM being fine-tuned; action = next token to generate; state = prompt + tokens
generated so far; reward = scalar from the reward model, applied only at the final token (end of
sequence), with intermediate steps having reward $0$ unless a process reward model is used.

**C.** Policy = the SFT model (frozen); action = choosing between two candidate responses;
state = prompt; reward = 1 if the chosen response is preferred, 0 otherwise.

**D.** Policy = the tokeniser; action = BPE merge operation; state = raw character sequence;
reward = compression ratio.

---

### Q6 — KL penalty in RLHF

The RLHF objective includes a KL divergence penalty:

$$\mathcal{L} = \mathbb{E}\left[r_\theta(x, y) - \beta \cdot D_{\text{KL}}\left(\pi_{\text{RL}} \| \pi_{\text{SFT}}\right)\right]$$

What is the purpose of the $\beta \cdot D_{\text{KL}}$ term?

**A.** It encourages the RL policy to match the reward model's output distribution.

**B.** It penalises the RL policy for diverging too far from the SFT reference policy, preventing
the policy from collapsing to degenerate outputs (repetition, nonsense) that exploit the reward
model's blind spots while still achieving high reward scores.

**C.** It maximises the entropy of the output distribution to encourage diversity.

**D.** It is equivalent to weight decay and serves only as a regulariser for the model weights.

---

### Q7 — Effect of $\beta$ in KL penalty

In the RLHF KL penalty term $\beta \cdot D_{\text{KL}}(\pi_{\text{RL}} \| \pi_{\text{SFT}})$,
what happens as $\beta \to 0$ and as $\beta \to \infty$?

**A.** $\beta \to 0$: policy collapses to a single token; $\beta \to \infty$: policy outputs
uniform noise.

**B.** $\beta \to 0$: KL penalty vanishes, and the policy is free to maximise reward without
constraints, increasing the risk of reward hacking; $\beta \to \infty$: the KL penalty
dominates and the RL policy is forced to stay very close to the SFT baseline, producing almost
no behavioural change from RL.

**C.** $\beta \to 0$: the policy converges faster; $\beta \to \infty$: the policy requires
more PPO steps.

**D.** $\beta$ has no effect on the final policy because it cancels out in the gradient update.

---

### Q8 — Reward hacking

What is "reward hacking" in the context of RLHF?

**A.** A technique used by annotators to inflate reward model scores for high-quality responses.

**B.** The phenomenon where the RL policy discovers input patterns or response styles that
achieve high scores from the reward model but do not actually correspond to genuinely helpful
or high-quality outputs -- exploiting the mismatch between the proxy reward and true human
preferences.

**C.** A hardware vulnerability that allows an adversary to modify reward model weights.

**D.** The process of manually adjusting reward model scores after training to correct errors.

---

### Q9 — Direct Preference Optimisation (DPO) objective

DPO (Rafailov et al., 2023) derives an objective directly from human preference data without
training a separate reward model.  The DPO loss is:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

What does this loss encourage?

**A.** It maximises the probability of both $y_w$ and $y_l$ relative to the reference policy.

**B.** It increases the probability of the preferred response $y_w$ relative to the reference
policy while decreasing the probability of the dispreferred response $y_l$ relative to the
reference policy, using the log-ratio as an implicit reward signal.

**C.** It minimises the KL divergence between the policy and the reward model.

**D.** It fine-tunes only the last layer of the model to adjust output probabilities.

---

### Q10 — DPO vs RLHF: practical differences

Compared to RLHF with PPO, what are the main practical advantages of DPO?

**A.** DPO requires more compute but achieves better alignment on all benchmarks.

**B.** DPO is simpler to implement (no RL loop, no separate reward model, no KL penalty
coefficient tuning), more stable to train (supervised loss on a fixed dataset), and uses less
GPU memory (no need to keep the reward model in memory during training).  The trade-off is
that DPO cannot incorporate online feedback and may be less adaptive than PPO.

**C.** DPO allows the use of unlabelled data, whereas RLHF requires annotated preference pairs.

**D.** DPO converges faster because it uses second-order optimisation.

---

### Q11 — Constitutional AI (CAI)

Constitutional AI (Bai et al., 2022, Anthropic) is an RLAIF approach.  Which description is
most accurate?

**A.** A human committee defines a legal "constitution" that the model must memorise verbatim.

**B.** CAI uses a set of natural-language principles (the "constitution") to guide a critic
model that revises model outputs and generates preference labels.  In the SL-CAI phase, the
model critiques and revises its own responses according to the principles; in the RLAIF phase,
a preference model trained on AI-generated labels replaces human annotators.  This reduces the
need for human labelling of harmful content.

**C.** CAI trains the model on a corpus of legal documents to make it compliant with regulations.

**D.** CAI is a hardware-level safety mechanism that intercepts the model's output logits before
sampling.

---

### Q12 — RLAIF vs RLHF

Reinforcement Learning from AI Feedback (RLAIF) substitutes an AI model for human annotators
in generating preference labels.  What is the primary motivation?

**A.** AI-generated labels are always more accurate than human labels.

**B.** Human annotation is expensive, slow, and exposes annotators to potentially harmful
content.  An AI labeller can generate preference data at much lower cost and at scale, enabling
iteration on alignment pipelines that would be impractical with purely human feedback.

**C.** RLAIF trains faster because it does not require backpropagation.

**D.** Human annotators cannot distinguish high-quality responses; AI labellers are better
calibrated to perplexity metrics.

---

### Q13 — Annotator disagreement in RLHF

Human preference annotations are often noisy and inconsistent.  How does the Bradley-Terry
model handle this?

**A.** Disagreeing annotations are discarded, and only unanimous preferences are used.

**B.** The Bradley-Terry model treats preferences as probabilistic rather than deterministic.
The reward model is trained to estimate the probability that one response is preferred over
another.  Noisy labels contribute to a softer gradient signal rather than causing hard
contradictions, and the model learns a calibrated preference score over a large enough dataset.

**C.** The model averages the reward scores across all annotators for each comparison.

**D.** Disagreements are resolved by asking a third annotator to break the tie; ties are
discarded.

---

### Q14 — SFT overfitting risk

A team fine-tunes a large LM on 1,000 high-quality instruction-response pairs using standard
next-token prediction loss.  What risk do they face, and what is the standard mitigation?

**A.** Risk: catastrophic forgetting of pretraining knowledge.  Mitigation: freeze all layers
and only train a small adapter head.

**B.** Risk: with a small dataset, the model can overfit -- memorising the training examples and
losing generalisation.  Additionally, computing loss on instruction tokens (not just response
tokens) can dilute the gradient signal.  Mitigation: use a moderate learning rate, a small
number of epochs (1--3), and mask the loss on prompt tokens.

**C.** Risk: the model learns to output shorter responses to minimise loss.  Mitigation: add a
length penalty to the loss.

**D.** Risk: SFT on fewer than 10,000 examples always fails; there is no mitigation.

---

### Q15 — PPO clipping objective

PPO uses a clipped surrogate objective to prevent large policy updates.  What does the clipping do?

**A.** It prevents the reward from exceeding a maximum value of 1.

**B.** It clips the probability ratio $\rho_t = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$
to the range $[1 - \epsilon, 1 + \epsilon]$ before multiplying by the advantage.  This prevents
excessively large policy updates in a single step, improving stability without requiring a
computationally expensive KL constraint.

**C.** It clips the gradient norm after the backward pass, equivalent to gradient clipping.

**D.** It truncates the episode length to $\epsilon$ steps, preventing very long rollouts.

---

### Q16 — Iterative preference data collection

Online RLHF (as opposed to offline DPO) collects new preference data from the evolving policy
during training.  Why does this matter?

**A.** Online data collection reduces GPU memory requirements.

**B.** As the policy improves, it generates responses in a different distribution than the
initial SFT model.  Reward model accuracy degrades on out-of-distribution samples.  Online
data collection ensures the reward model and human annotators evaluate responses from the
current policy, keeping the training on-distribution and improving the reward model's coverage.

**C.** Online collection is required because the reward model cannot evaluate text that was not
seen during its training.

**D.** Online RLHF is strictly worse than offline DPO; it is only used for historical reasons.

---

### Q17 — DPO with reference policy

The DPO loss requires a reference policy $\pi_{\text{ref}}$.  What is this reference policy
and why is it necessary?

**A.** The reference policy is the reward model; its scores provide the training signal.

**B.** The reference policy is typically the SFT model (frozen).  It serves as the anchor that
defines the "implicit reward" $r^* = \beta \log (\pi_\theta / \pi_{\text{ref}})$.  Without a
reference, the loss would only push up $\pi_\theta(y_w)$ and down $\pi_\theta(y_l)$ without
any constraint, risking degenerate solutions where the model collapses to high probability on
$y_w$ templates regardless of the prompt.

**C.** The reference policy is a randomly initialised model that provides diverse gradient signal.

**D.** The reference policy is trained jointly with the DPO policy in the same optimisation step.

---

### Q18 — Evaluation of aligned models

The "alignment tax" refers to a trade-off sometimes observed after alignment fine-tuning.
What does this term mean?

**A.** Alignment fine-tuning makes the model more expensive to run at inference time.

**B.** Alignment fine-tuning (especially RLHF) can slightly reduce performance on certain
capability benchmarks (e.g., coding, mathematical reasoning) compared to the base model, as the
model learns to favour safe, helpful responses over maximising raw task performance.  More
recent work (e.g., strong RLHF pipelines, DPO with high-quality data) has largely reduced but
not fully eliminated this tax.

**C.** Aligned models charge API users more per token to cover the cost of human annotators.

**D.** The alignment tax is a legal concept referring to compliance costs for AI companies.

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
| 12 | B      |
| 13 | B      |
| 14 | B      |
| 15 | B      |
| 16 | B      |
| 17 | B      |
| 18 | B      |

---

## Detailed Explanations

---

### Q1 — Purpose of alignment fine-tuning

**Correct: B.**

Pretraining maximises $P(x_t \mid x_{<t})$ over a large web corpus.  This teaches fluent text
generation but also imitation of harmful, biased, or low-quality content present in training
data.  A pretrained model will "helpfully" complete prompts in ways that might be harmful
because it was trained to predict what comes next on the web, not to be helpful.  SFT and
RLHF teach the model to follow user intent, decline harmful requests, and produce high-quality
responses.

- **A is wrong.** Pretrained models generate text perfectly well without alignment; the issue
  is the QUALITY and SAFETY of what they generate.
- **C is wrong.** Alignment fine-tuning does not reduce parameter count.
- **D is wrong.** Autoregressive generation is a property of the architecture and decoding
  algorithm, not of the training objective.

---

### Q2 — SFT data format

**Correct: B.**

Computing loss only on assistant response tokens is standard practice (sometimes called "response
masking").  Including loss on instruction tokens would teach the model to predict the instruction
itself, diluting gradient signal and potentially causing the model to generate instruction-like
outputs during response generation.

- **A is wrong.** While the underlying loss is next-token prediction, the data format is
  structured as instruction-response pairs with response-only loss masking.
- **C is wrong.** Document-summary pairs are used for summarisation fine-tuning but are not the
  general SFT format for instruction following.
- **D is wrong.** SFT data includes positive examples of good responses, not empty refusals.

---

### Q3 — Reward model training

**Correct: B.**

The reward model is initialised from the SFT checkpoint (which already produces sensible text)
and an additional linear head is added to produce a scalar.  Training uses the Bradley-Terry
pairwise comparison objective.  The choice of annotations as pairwise comparisons (rather than
absolute ratings) is deliberate: humans are more consistent at ranking responses relatively
than assigning absolute quality scores.

- **A is wrong.** Training from scratch would be inefficient and ignores the representations
  already learned during pretraining.
- **C is wrong.** The reward model is trained on human annotations; the RL policy does not
  train the reward model.
- **D is wrong.** The standard reward model outputs a single scalar, not an ensemble of binary
  classifiers.

---

### Q4 — Bradley-Terry reward loss

**Correct: B.**

The sigmoid $\sigma(r_w - r_l)$ is the Bradley-Terry probability that $y_w$ is preferred.
Maximising its log is equivalent to minimising the negative log-likelihood of the observed
human preference under this model.  As $r_w - r_l$ increases, $\sigma \to 1$ and the loss
approaches 0; as the difference decreases or reverses, the loss increases, pushing the reward
model to widen the gap.

- **A is wrong.** The loss does not constrain the absolute reward values; only the relative
  ranking is trained.
- **C is wrong.** MSE would require absolute ground-truth ratings, which are noisier than
  relative rankings.
- **D is wrong.** The scalar reward is unbounded; the sigmoid is applied to the DIFFERENCE, not
  the raw reward.

---

### Q5 — PPO formulation in RLHF

**Correct: B.**

In the RLHF MDP formulation, the LM is the policy, tokens are actions, and the state is
the sequence generated so far.  The episode terminates when an end-of-sequence token is
produced.  The reward signal from the reward model is typically a single scalar $r(x, y)$
applied at the final token.  This "bandit" formulation (reward only at the end) is one reason
RLHF can be unstable; process reward models (PRMs) assign reward at each step but are harder
to train.

- **A is wrong.** The policy is the LM being optimised, not the reward model.
- **C is wrong.** The SFT model is the REFERENCE policy (frozen baseline for KL), not the policy
  being updated.
- **D is wrong.** The tokeniser is a deterministic preprocessing step, not a policy.

---

### Q6 — KL penalty purpose

**Correct: B.**

The reward model is an imperfect proxy trained on a limited set of comparison data.  Without
the KL constraint, PPO can find degenerate responses (e.g., repetitive praise, verbose
hedging, or adversarial patterns) that fool the reward model into assigning high scores.
The KL penalty prevents the policy from moving far from the SFT baseline, anchoring it in the
region where the reward model's estimates are reliable.

- **A is wrong.** The KL is between the RL policy and the SFT policy, not involving the reward
  model's distribution.
- **C is wrong.** Entropy maximisation is a different regularisation technique; the KL penalty
  reduces rather than increases the policy's distance from the SFT reference.
- **D is wrong.** The KL is over output distributions, not model weights; it is fundamentally
  different from L2 weight decay.

---

### Q7 — Effect of $\beta$

**Correct: B.**

$\beta$ controls the strength of the KL anchor:
- $\beta \to 0$: unconstrained reward maximisation; the policy can move anywhere in distribution
  space and will exploit the reward model.
- $\beta \to \infty$: the KL penalty dominates; any move away from the SFT policy is so costly
  that the policy effectively stays at $\pi_{\text{SFT}}$, achieving no alignment benefit from RL.
Typical values are in the range $\beta \in [0.01, 0.5]$, balancing exploration and stability.

- **A is wrong.** Neither extreme causes uniform-noise generation; the policy stays in the
  language model distribution.
- **C is wrong.** $\beta$ affects the objective landscape, not the convergence speed of PPO
  directly.
- **D is wrong.** $\beta$ appears in the objective and directly shapes the gradient.

---

### Q8 — Reward hacking

**Correct: B.**

Classic examples: models learn to produce very long responses (reward models often prefer
longer, more detailed answers, regardless of quality), to add excessive caveats and disclaimers
(which look "safe" to annotators), or to use confident assertive language (which sounds
authoritative).  Gao et al. (2022) formalised reward hacking in terms of "gold reward vs.
proxy reward" divergence, showing it scales with the number of RL training steps.

- **A is wrong.** Annotators do not "hack" scores; this describes a model behaviour.
- **C is wrong.** Reward hacking is a training dynamics phenomenon, not a hardware attack.
- **D is wrong.** Manual score adjustment post-training is a different practice (score calibration).

---

### Q9 — DPO objective

**Correct: B.**

The DPO loss is derived from the RLHF objective by analytically solving for the optimal policy
and substituting back.  The implicit reward for response $y$ given context $x$ under the
optimal policy is $\hat{r}(x, y) = \beta \log (\pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x))$.
DPO plugs this into the Bradley-Terry loss, giving a loss that increases the log-ratio of the
winning response over the reference and decreases it for the losing response.

- **A is wrong.** The DPO loss has opposite signs for $y_w$ and $y_l$; it cannot increase both
  simultaneously.
- **C is wrong.** No reward model is trained or used in DPO.
- **D is wrong.** DPO fine-tunes all parameters of the model through standard gradient descent;
  it does not restrict updates to the last layer.

---

### Q10 — DPO vs RLHF practical differences

**Correct: B.**

RLHF with PPO requires: (1) training a reward model, (2) running a PPO training loop with 4
models in memory simultaneously (policy, reference, reward, value function), (3) collecting
rollouts, and (4) tuning KL coefficient, clip ratio, and PPO hyperparameters.  DPO reduces
this to a single supervised fine-tuning step on preference data with a custom loss.  The
simplicity gain is substantial, which is why DPO and its variants (IPO, KTO, SimPO) have been
widely adopted.

- **A is wrong.** DPO typically requires LESS compute than RLHF with PPO.
- **C is wrong.** DPO still requires preference pair labels; it does not use unlabelled data.
- **D is wrong.** DPO uses standard first-order gradient descent (Adam).

---

### Q11 — Constitutional AI

**Correct: B.**

CAI operates in two phases.  SL-CAI: the model is shown a harmful prompt, generates a
response, critiques it according to a constitutional principle (e.g., "this response may be
harmful -- revise it to be helpful and harmless"), and the revised response is used as a
training target.  RLAIF: a feedback model assigns preferences between original and revised
responses, generating AI-labelled preference data for a preference model, which then serves as
a reward signal for RL fine-tuning.  This dramatically reduces the volume of human annotations
required for the harmlessness dimension.

- **A is wrong.** The "constitution" is a set of natural-language principles, not a legal
  document to memorise.
- **C is wrong.** The constitution consists of ethical principles, not legal statutes.
- **D is wrong.** CAI is a training procedure, not a hardware or logit-level filter.

---

### Q12 — RLAIF motivation

**Correct: B.**

Human annotation for RLHF requires: sourcing a diverse annotator pool, training annotators,
managing quality control, and protecting annotators from disturbing content.  At scale (billions
of comparisons), this is infeasible.  AI labellers (Claude, GPT-4, etc.) can generate
preference data at much larger scale.  The key risk is "AI feedback bias": if the AI labeller
has its own biases or preferences, these are amplified in the trained model.

- **A is wrong.** AI labels are not always more accurate; they can reflect systematic biases
  of the labelling model.
- **C is wrong.** RLAIF still uses backpropagation; it only substitutes the label source.
- **D is wrong.** Human annotators are generally good at quality assessment; the bottleneck is
  scale and cost, not calibration.

---

### Q13 — Annotator disagreement

**Correct: B.**

The Bradley-Terry model is inherently probabilistic: $P(y_w \succ y_l) = \sigma(r_w - r_l)$.
With 60% of annotators preferring $y_w$, the reward model targets $P \approx 0.6$, which
corresponds to a modest positive reward difference.  This is more principled than discarding
or aggregating, and it is statistically consistent: with enough data, the learned reward model
will accurately reflect the population's distribution of preferences.

- **A is wrong.** Discarding disagreements would discard a majority of real-world preference
  data (inter-annotator agreement is typically 60--75% even among well-trained annotators).
- **C is wrong.** Averaging absolute scores requires consistent scales across annotators, which
  is not guaranteed; pairwise comparisons are more robust.
- **D is wrong.** Tie-breaking by a third annotator is sometimes done in practice but is not
  how the reward model objective handles noise mathematically.

---

### Q14 — SFT overfitting risk

**Correct: B.**

Standard practice for SFT is 1--3 epochs with a learning rate 1--2 orders of magnitude lower
than pretraining, and response masking (computing loss only on the assistant turns).  Overfitting
on small SFT datasets is a real failure mode: the model may repeat phrasing from training
examples verbatim or generate responses only for prompt patterns similar to those in the
training set.

- **A is wrong.** Catastrophic forgetting is a risk when fine-tuning at a high learning rate
  for many epochs, but the mitigation is a low learning rate, not freezing all layers.
- **C is wrong.** SFT models do not systematically produce shorter responses; this is not a
  documented failure mode of standard SFT.
- **D is wrong.** Many high-quality instruction-following models have been SFT fine-tuned on
  hundreds to a few thousand examples (e.g., Alpaca used 52K, but LIMA (Zhou et al., 2023)
  used only 1K).

---

### Q15 — PPO clipping

**Correct: B.**

PPO-Clip replaces the KL-constrained trust region update of TRPO with the simpler clipping
mechanism.  By bounding the probability ratio, it ensures the new policy cannot move too far
from the old one in a single gradient step.  This makes PPO far more computationally tractable
than TRPO while retaining most of its stability benefits.

- **A is wrong.** Clipping is applied to the probability ratio, not the reward.
- **C is wrong.** Gradient clipping is a separate operation applied to gradient norms; PPO
  clipping is applied to the policy ratio in the objective.
- **D is wrong.** Episode length in RLHF for LMs is determined by the generation length, not
  by $\epsilon$.

---

### Q16 — Online vs offline RLHF

**Correct: B.**

This is the standard "distribution shift" problem in RL.  A reward model trained on SFT-model
outputs may generalise poorly to outputs from a more capable RL-trained policy.  Online RL
(sampling from the current policy, obtaining reward/feedback, updating) keeps training
on-distribution.  Offline methods like DPO operate on a fixed dataset and can struggle when the
policy's distribution moves far from the data collection policy.

- **A is wrong.** Online data collection increases compute cost, not reduces memory.
- **C is wrong.** Reward models generalise to any text; the issue is accuracy degradation on
  out-of-distribution text, not a hard inability to score.
- **D is wrong.** Both online RLHF and offline DPO have distinct trade-offs; neither is strictly
  superior in all settings.

---

### Q17 — DPO reference policy

**Correct: B.**

The DPO derivation shows that the optimal policy under the RLHF objective takes the form
$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \exp(r^*(x, y) / \beta)$.
Rearranging: $r^*(x, y) = \beta \log(\pi^*(y \mid x) / \pi_{\text{ref}}(y \mid x)) + \beta \log Z(x)$.
The reference policy anchors what "neutral" looks like; without it, the log-ratio is undefined.
Using the SFT model as reference ensures the policy stays in the neighbourhood of fluent,
instruction-following outputs.

- **A is wrong.** In DPO there is no explicit reward model; the reference policy is the SFT
  model.
- **C is wrong.** A random reference policy would produce meaningless log-ratios and unstable
  training.
- **D is wrong.** The reference policy is frozen during DPO training; updating it would change
  the anchor every step, destabilising the objective.

---

### Q18 — Alignment tax

**Correct: B.**

The alignment tax was documented in the original InstructGPT paper: RLHF-tuned models sometimes
scored lower on standard NLP benchmarks (TruthfulQA, WinogradNLI) than the base model.
The model learns to hedge, qualify, and be verbose, which can hurt precision-requiring tasks.
Modern approaches (Constitutional AI, strong DPO with diverse data, careful reward model
design) have substantially narrowed this gap; models like Claude 3 and GPT-4 show minimal or
no alignment tax on most benchmarks.

- **A is wrong.** Alignment fine-tuning does not change the model architecture or inference
  cost; it only changes the weights.
- **C is wrong.** The alignment tax is a technical ML concept, not a pricing or business term.
- **D is wrong.** This is also a misuse of the term; the alignment tax is a capability
  degradation phenomenon, not a legal or financial concept.
