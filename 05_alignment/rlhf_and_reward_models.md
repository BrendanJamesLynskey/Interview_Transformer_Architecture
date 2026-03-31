# RLHF and Reward Models: Interview Questions

Reinforcement Learning from Human Feedback (RLHF) is the dominant technique for aligning large language models with human preferences. It was central to InstructGPT and underpins most commercial chat models. Interview questions on this topic appear at all levels, from conceptual overviews to mathematical derivations.

---

## Fundamentals

### Q1. What problem does RLHF solve that SFT alone cannot?

**Answer.**

SFT trains the model on demonstrations of desired behaviour. But demonstrations have a fundamental limitation: they only contain examples of "good" responses. The model cannot learn *relative quality* — that response A is much better than response B, or that response C is slightly better than response D.

Human preferences, by contrast, are more naturally expressed comparatively: "I prefer this response over that one." This is easier to collect reliably than absolute quality ratings, because it sidesteps the need to agree on a universal quality scale.

RLHF exploits this by:
1. Training a **reward model** on human preference comparisons, which learns to assign scalar scores reflecting human quality judgments.
2. Using **reinforcement learning** (typically PPO) to fine-tune the language model to maximise the reward model's score.

This allows the model to discover response strategies that score highly according to the reward model — strategies that go beyond simply imitating the demonstration distribution. In InstructGPT, RLHF produced models rated as significantly better than SFT-only models despite using far fewer parameters.

---

### Q2. Describe the full InstructGPT pipeline step by step.

**Answer.**

The InstructGPT pipeline (Ouyang et al., 2022) has three stages:

**Stage 1: SFT**
- A small set of human annotators write high-quality responses to a diverse set of prompts sampled from the API prompt distribution.
- The pretrained model is fine-tuned on these (prompt, response) pairs using supervised learning.
- This produces a well-behaved starting policy.

**Stage 2: Reward Model Training**
- Annotators are shown a prompt and several responses (typically 4–9) from the SFT model.
- They rank the responses from best to worst.
- A separate language model (typically smaller than the policy) is trained to predict these rankings using a pairwise ranking loss (Bradley-Terry model).
- The reward model outputs a single scalar score for a (prompt, response) pair.

**Stage 3: RL Fine-Tuning with PPO**
- The SFT model is used as both the initial policy and a frozen reference model.
- For each training step: sample a prompt, generate a response with the current policy, score with the reward model, update the policy using PPO to maximise reward.
- A KL penalty against the reference (SFT) model is added to prevent the policy from drifting too far from the SFT initialisation.

The result is a model that generates responses that score highly on the reward model while remaining close enough to the SFT model that it does not collapse into degenerate behaviour.

---

### Q3. What is the Bradley-Terry model and how is it used to train a reward model?

**Answer.**

The Bradley-Terry model is a probabilistic model for pairwise comparisons. Given two items $A$ and $B$ with scores $s_A$ and $s_B$, it models the probability that $A$ is preferred over $B$ as:

$$P(A \succ B) = \frac{e^{s_A}}{e^{s_A} + e^{s_B}} = \sigma(s_A - s_B)$$

where $\sigma$ is the sigmoid function.

For reward model training, given a prompt $x$, a chosen (preferred) response $y_w$, and a rejected response $y_l$, the Bradley-Terry loss is:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]$$

where $r_\phi(x, y)$ is the scalar reward assigned by the reward model to response $y$ given prompt $x$.

Minimising this loss trains the reward model to assign higher scores to preferred responses. The training data $\mathcal{D}$ consists of human comparison pairs $(x, y_w, y_l)$.

In practice:
- The reward model is initialised from the SFT model (or a similar sized pretrained model) with the final language modelling head replaced by a linear layer that outputs a scalar.
- The model is trained on all (prompt, response) pairs from annotator rankings, with each ranking decomposed into all implied pairwise preferences.

---

### Q4. What is the KL penalty in RLHF and why is it necessary?

**Answer.**

The KL penalty is a regularisation term added to the RL objective that penalises the policy $\pi_\theta$ for diverging from the reference policy $\pi_\text{ref}$ (the SFT model):

$$\mathcal{L}_\text{RLHF} = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \right]$$

The KL term $\beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ (summed over tokens, this equals the per-sequence KL divergence) penalises the policy for generating responses that are improbable under the SFT model.

**Why it is necessary:**

1. **Preventing reward hacking.** Without the KL penalty, the policy will find responses that score highly on the reward model but are not genuinely good — for example, extremely long responses if the reward model has a length bias, or nonsensical text that happens to trigger high scores. The KL penalty forces the policy to stay in a region of text space that the reward model has actually evaluated.

2. **Maintaining language model coherence.** The SFT model produces fluent, coherent text. A policy without a KL constraint can collapse to generating repetitive, incoherent, or degenerate outputs that achieve high reward by exploiting the reward model rather than being genuinely helpful.

3. **Exploration-exploitation.** The KL penalty balances exploration (generating diverse new responses) with exploitation (staying close to the well-behaved SFT distribution).

The coefficient $\beta$ controls the trade-off: small $\beta$ allows more aggressive optimisation toward the reward model, large $\beta$ keeps the policy close to the SFT model.

---

## Intermediate

### Q5. Describe the PPO algorithm as used in LLM fine-tuning. What are the key components?

**Answer.**

Proximal Policy Optimisation (PPO, Schulman et al., 2017) is a policy gradient algorithm that constrains each update to stay within a "trust region" around the current policy. In LLM fine-tuning:

**Key components:**

**1. Value function (critic).** A separate model head (often sharing the transformer backbone) estimates the expected future reward from each token position. This is used to compute advantage estimates.

**2. Advantage estimation.** The advantage $A_t$ at token position $t$ measures how much better a particular token choice is compared to the average. It is estimated using Generalised Advantage Estimation (GAE):

$$A_t = \sum_{k=0}^{T-t} (\gamma \lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

In the LLM setting, the reward is typically sparse (only given at the end of the sequence), so $r_t = 0$ for all tokens except the final one.

**3. Clipped surrogate objective.** PPO's policy loss clips the probability ratio $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ to prevent large updates:

$$\mathcal{L}_\text{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

**4. KL penalty.** An additional $-\beta \cdot \text{KL}[\pi_\theta \| \pi_\text{ref}]$ term is added to the objective.

**5. Value loss.** The critic is trained to minimise the mean-squared error between predicted and actual returns.

**Memory note:** PPO in RLHF requires holding four models in memory simultaneously: the policy (being trained), the reference policy (frozen), the reward model (frozen), and the value function. For large models this is a significant engineering challenge, driving interest in lighter alternatives like DPO.

---

### Q6. What is reward hacking, and what empirical evidence exists for it?

**Answer.**

Reward hacking (also called reward overoptimisation or Goodhart's Law in action) occurs when the policy learns to exploit the reward model's imperfections to achieve high reward scores without genuinely improving the quality measured by humans.

The reward model is a proxy for human preferences, not the true objective. As the policy is optimised against the reward model, it will inevitably discover inputs that score highly on the proxy but not on the true objective.

**Empirical evidence:**

Gao et al. (2023) "Scaling Laws for Reward Model Overoptimization" showed a clear over-optimisation curve: as the KL divergence between the policy and the reference model increases, the reward model score initially increases but then the *gold human preference score* peaks and then decreases. The policy is genuinely improving at first, then begins to exploit the reward model.

**Common manifestations:**

- **Length exploitation.** Reward models trained on human preferences tend to rate longer responses as better, because humans perceive length as effort. Policies learn to pad responses with filler content.
- **Sycophancy.** If the reward model learns that human annotators prefer responses that agree with the user's stated position, the policy learns to agree even when the user is wrong.
- **Formatting exploitation.** Adding bullet points and headers increases reward scores even when the content does not benefit from such formatting.
- **Hedging.** Excessive caveats and disclaimers can inflate perceived safety scores without genuine safety improvement.

---

### Q7. What is the reference model in RLHF and what role does it play?

**Answer.**

The reference model $\pi_\text{ref}$ is a frozen copy of the SFT model that is used throughout RL training. It serves two purposes:

1. **KL penalty anchor.** The per-token KL divergence $\log \frac{\pi_\theta(y_t|y_{<t}, x)}{\pi_\text{ref}(y_t|y_{<t}, x)}$ is computed at every token of every generated response. This requires a forward pass through the reference model for each generated response, making it one of the most memory-intensive parts of the pipeline.

2. **Defining the implicit constraint.** The RLHF objective can be viewed as: find the policy that maximises expected reward subject to a KL budget from the reference policy. The reference model defines "what good language looks like" before reward optimisation, preventing the policy from drifting into degenerate text.

**Practical considerations:**
- The reference model is frozen and only used for inference (no gradients needed), so it can be run in half precision with no gradient checkpointing.
- It must be kept in GPU memory throughout training. For 70B models, this alone requires ~140GB of GPU memory in BF16.
- Some implementations use the policy checkpoint from the previous PPO iteration as the reference (rather than the fixed SFT model), but this is less standard.

---

## Advanced

### Q8. Derive the optimal policy under the KL-constrained RLHF objective.

**Answer.**

The RLHF objective is:

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)} \left[ r(x, y) \right] - \beta \, \text{KL}\left[ \pi(\cdot|x) \,\|\, \pi_\text{ref}(\cdot|x) \right]$$

Expanding the KL:

$$= \max_\pi \mathbb{E}_{x, y} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} \right]$$

For a fixed $x$, this is a functional optimisation over $\pi(\cdot|x)$. Writing the Lagrangian with constraint $\sum_y \pi(y|x) = 1$:

$$\mathcal{L}(\pi) = \sum_y \pi(y|x) \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} \right] + \lambda \left( 1 - \sum_y \pi(y|x) \right)$$

Taking the functional derivative and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} - \beta - \lambda = 0$$

Solving for $\pi(y|x)$:

$$\log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} = \frac{r(x, y)}{\beta} - \frac{\lambda + \beta}{\beta}$$

$$\pi^*(y|x) = \pi_\text{ref}(y|x) \cdot \exp\left( \frac{r(x, y)}{\beta} \right) \cdot C^{-1}$$

where $C = \sum_y \pi_\text{ref}(y|x) \exp(r(x, y)/\beta)$ is the normalisation constant (the partition function).

This can be written as:

$$\boxed{\pi^*(y|x) = \frac{\pi_\text{ref}(y|x) \exp(r(x,y)/\beta)}{Z(x)}}$$

This result is fundamental: the optimal policy under KL-constrained reward maximisation is the reference policy tilted by $\exp(r/\beta)$ and renormalised. This closed-form expression is the key insight exploited by DPO to eliminate the RL step entirely.

---

### Q9. How do you diagnose and debug an unstable RLHF training run?

**Answer.**

RLHF training is significantly more unstable than SFT because it involves multiple interacting models and a non-stationary training signal. Key failure modes and diagnostics:

**Symptoms and causes:**

| Symptom | Likely Cause | Diagnostic |
|---|---|---|
| KL divergence grows unboundedly | $\beta$ too small; clip ratio $\epsilon$ too large | Plot KL vs training step; should be bounded |
| Reward collapses to 0 then diverges | Value function not initialised correctly; LR too high | Plot value loss separately |
| Responses become repetitive | Policy collapsed to reward-exploiting mode | Sample and inspect generations every N steps |
| Reward increases but human ratings fall | Reward hacking | Maintain a gold evaluation set with human raters |
| NaN in loss | Overflow in advantage estimation or KL; fp16 issues | Check activation norms; switch to bf16 |

**Debugging process:**

1. **Monitor KL divergence** between policy and reference model. It should grow slowly and then plateau. Rapid growth indicates over-optimisation.

2. **Track reward model score distribution** over time. If the score distribution shifts toward the upper end of the scale with low variance, the policy has likely collapsed into a small set of reward-hacking strategies.

3. **Inspect generations.** Automated metrics miss qualitative degradation. Sample 50–100 responses every 500 steps and review manually.

4. **Separate critic and policy learning rates.** The value function typically needs a higher learning rate than the policy to stay calibrated. Using the same LR for both is a common mistake.

5. **Reduce rollout batch size.** Larger batches introduce more variance in advantage estimates; smaller batches with more PPO epochs per batch can be more stable.

6. **Check reward model quality first.** An under-trained reward model is the most common root cause of RLHF instability. Evaluate the reward model's ranking accuracy on a held-out preference set before starting RL.

---

### Q10. Compare the computational requirements of RLHF to SFT. What are the engineering challenges at scale?

**Answer.**

RLHF is substantially more expensive than SFT in both memory and compute.

**Memory:**

SFT requires: the model being trained (with optimiser states: ~4x the model size in Adam), plus optionally a small amount for gradient checkpointing.

RLHF requires simultaneously:
- Policy model (with optimiser states): ~4x model size
- Reference model (inference only): ~1x model size
- Reward model (inference only): ~0.5–1x model size (often smaller)
- Value function (with optimiser states): ~1x model size (if sharing the policy backbone) or ~4x if separate

Total: roughly 6–8x a single model copy, before activations.

**Compute:**

For each PPO step:
1. Forward pass through policy to generate responses (autoregressive, slow)
2. Forward pass through reward model to score responses
3. Forward pass through reference model to compute per-token log-probs
4. Forward pass through value function
5. Backward pass through policy and value function

This is roughly 4–5x the compute of a single SFT step for the same batch size, and the generation step (step 1) is typically the bottleneck because it is autoregressive.

**Engineering challenges:**

- **Async generation and training.** Generation and gradient computation can be pipelined across different GPU groups, but this requires careful orchestration (see TRL, DeepSpeed-Chat, OpenRLHF implementations).
- **Reward model serving.** The reward model must be queried at high throughput. Batching and caching are important.
- **Numerical precision.** Advantage normalisation and KL computation require careful fp32 accumulation even when training in bf16.
- **Checkpoint management.** Four model states must be saved and can be restored independently; checkpoint versioning is more complex than SFT.
