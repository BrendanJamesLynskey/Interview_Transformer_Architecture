# Worked Problem 02: DPO Loss Derivation and Comparison with RLHF

**Topic:** Step-by-step derivation of the DPO loss from the RLHF objective. Computational comparison between DPO and RLHF.

**Difficulty:** Advanced

**Prerequisites:** Problem 01 (RLHF derivation), Bradley-Terry model, log-likelihood maximisation.

---

## Problem Statement

1. Starting from the RLHF optimal policy derived in Problem 01, derive the DPO loss function.
2. Explain at each step what mathematical transformation is performed and why it is valid.
3. Compare the computational requirements of DPO and RLHF (PPO) for a 70B parameter model.
4. Identify the key assumptions that DPO makes and when they break down.

---

## Solution

### Part 1: Full Derivation of the DPO Loss

**Step 1: Recall the RLHF optimal policy.**

From Problem 01, the optimal policy under the KL-constrained reward maximisation objective is:

$$\pi^*(y|x) = \frac{\pi_\text{ref}(y|x) \cdot \exp(r(x,y)/\beta)}{Z(x)}$$

where $Z(x) = \sum_y \pi_\text{ref}(y|x) \exp(r(x,y)/\beta)$ is the partition function.

**Step 2: Rearrange to express the reward in terms of the policy.**

Take the logarithm of both sides:

$$\log \pi^*(y|x) = \log \pi_\text{ref}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)$$

Multiply through by $\beta$ and isolate $r(x,y)$:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

This is the **implicit reward**: the reward function is completely determined by the optimal policy and the reference policy. The partition function $\beta \log Z(x)$ acts as a prompt-dependent baseline.

**Step 3: Substitute into the Bradley-Terry preference model.**

Human preferences are modelled using the Bradley-Terry model. Given responses $y_w$ (preferred, "winner") and $y_l$ (dispreferred, "loser") for prompt $x$, the probability of the human preferring $y_w$ is:

$$P^*(y_w \succ y_l \mid x) = \sigma(r(x,y_w) - r(x,y_l))$$

Substituting the implicit reward expression for both responses:

$$r(x,y_w) - r(x,y_l) = \left[ \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)} + \beta \log Z(x) \right]$$

The partition function terms cancel:

$$= \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)}$$

Therefore:

$$P^*(y_w \succ y_l \mid x) = \sigma\!\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)} \right)$$

**Key observation:** the intractable partition function $Z(x)$ has cancelled. The preference probability depends only on log-ratios, which are tractable.

**Step 4: Replace the unknown optimal policy with the trainable policy.**

We do not have access to $\pi^*$ — that is what we are trying to learn. We parameterise the optimal policy with $\pi_\theta$ and maximise the likelihood of the observed preference data:

$$\max_\theta \mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log P_\theta(y_w \succ y_l \mid x) \right]$$

$$= \max_\theta \mathbb{E}_\mathcal{D} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

Converting to a minimisation (negating):

$$\boxed{\mathcal{L}_\text{DPO}(\pi_\theta; \pi_\text{ref}) = -\mathbb{E}_\mathcal{D} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]}$$

**This is the DPO loss.** It is a supervised loss requiring no RL, no value function, and no explicit reward model.

---

### Part 2: What Does Each Component Compute?

During a training step, for a batch of triplets $(x, y_w, y_l)$:

**Computing the log-ratios:**

$$\Delta_w = \log \pi_\theta(y_w|x) - \log \pi_\text{ref}(y_w|x)$$
$$\Delta_l = \log \pi_\theta(y_l|x) - \log \pi_\text{ref}(y_l|x)$$

where $\log \pi_\theta(y|x) = \sum_{t=1}^T \log \pi_\theta(y_t | y_{<t}, x)$ is the sum of per-token log-probabilities from the language model.

The implicit reward assigned to response $y$ is $\hat{r}(x,y) = \beta \cdot \Delta$.

**The loss:**

$$\mathcal{L} = -\log \sigma(\beta(\Delta_w - \Delta_l))$$

The gradient pushes toward increasing $\Delta_w$ (making $y_w$ more likely relative to the reference) and decreasing $\Delta_l$ (making $y_l$ less likely relative to the reference). The sigmoid weighting ensures that "easy" pairs (where $\Delta_w > \Delta_l$ already) contribute less gradient than "hard" pairs (where the model currently misranks the pair).

**Implementation sketch:**

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logprobs_w, policy_logprobs_l,
             ref_logprobs_w, ref_logprobs_l, beta=0.1):
    """
    Args:
        policy_logprobs_w: (batch,) sum of log-probs for y_w under policy
        policy_logprobs_l: (batch,) sum of log-probs for y_l under policy
        ref_logprobs_w:    (batch,) sum of log-probs for y_w under reference
        ref_logprobs_l:    (batch,) sum of log-probs for y_l under reference
        beta: KL regularisation coefficient
    Returns:
        Scalar DPO loss
    """
    log_ratio_w = policy_logprobs_w - ref_logprobs_w  # Delta_w
    log_ratio_l = policy_logprobs_l - ref_logprobs_l  # Delta_l
    reward_diff = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(reward_diff).mean()
    return loss
```

---

### Part 3: Computational Requirements Comparison

We compare training a 70B parameter model. Assume:
- BF16 precision: 2 bytes per parameter
- 70B parameters → 140 GB per model copy (weights only)
- Full Adam optimiser states: 3x extra (fp32 weights + first moment + second moment) = 420 GB for the model being trained

**DPO memory requirements:**

| Component | Memory |
|---|---|
| Policy model (BF16 weights) | 140 GB |
| Policy optimiser states (fp32) | 420 GB |
| Reference model (BF16, frozen, inference only) | 140 GB |
| Activations (gradient checkpointing) | ~10–30 GB |
| **Total** | **~720 GB** |

**RLHF/PPO memory requirements:**

| Component | Memory |
|---|---|
| Policy model (BF16 weights) | 140 GB |
| Policy optimiser states (fp32) | 420 GB |
| Reference model (BF16, frozen) | 140 GB |
| Reward model (assume 13B, BF16) | 26 GB |
| Value function (shared backbone with policy) | ~10 GB (extra head) |
| Value function optimiser states | ~30 GB |
| Activations + rollout buffers | ~30–60 GB |
| **Total** | **~820–900 GB** |

**Compute per training step:**

DPO requires:
1. Forward pass through policy on $(x, y_w)$ and $(x, y_l)$: 2 forward passes
2. Forward pass through reference model on $(x, y_w)$ and $(x, y_l)$: 2 forward passes (no grad)
3. Backward pass through policy: 1 backward pass
- Total: equivalent to ~3 full model forward passes (backward ≈ 2x forward)

RLHF/PPO requires per iteration:
1. Autoregressive generation (rollout): ~T forward passes for a response of length T (slow; serial)
2. Forward pass through reward model on generated response
3. Forward pass through reference model on generated response
4. Forward pass through value function
5. PPO backward pass through policy
6. Value function backward pass
- Total: significantly more than DPO, dominated by the autoregressive generation step

For responses of length 256 tokens, RLHF requires roughly 256 sequential forward steps per example in the rollout phase alone. DPO, operating on pre-generated responses stored in a dataset, avoids this entirely.

**Practical consequence:** DPO is typically 3–8x faster per training step than RLHF/PPO on the same hardware, and requires fewer GPU hours to reach comparable performance from a fixed preference dataset.

---

### Part 4: Key Assumptions in DPO and When They Break Down

**Assumption 1: The preference data was sampled from (or is close to) the reference policy.**

The derivation substitutes $\pi^*$ with $\pi_\theta$ and assumes the data distribution is compatible with the reference policy. If $y_w$ and $y_l$ were generated by a different (e.g., much stronger) model, the log-ratio $\log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ may diverge, and the implicit reward becomes uninformative.

*When it breaks down:* Using GPT-4-generated responses as $y_w$ and GPT-3.5 responses as $y_l$ while training a 7B model from a 7B SFT reference. The responses are far from the 7B model's distribution.

*Mitigation:* iterative DPO — regenerate response pairs from the current policy periodically.

**Assumption 2: The Bradley-Terry model accurately represents human preferences.**

DPO inherits the Bradley-Terry assumption: preferences are determined by a single scalar reward, and preferences are consistent and transitive ($A \succ B$ and $B \succ C$ implies $A \succ C$). Human preferences often violate this, especially when different dimensions of quality are traded off (e.g., a response might be more accurate but less concise than another).

*When it breaks down:* tasks where preferences are highly context-dependent, multidimensional, or non-transitive. For example, creative writing where "better" depends on subjective taste.

**Assumption 3: The preference labels are noise-free.**

DPO treats all preference labels as ground truth. In practice, human annotators agree only ~60–70% of the time on challenging preference pairs, and synthetic preference data has its own systematic biases.

*When it breaks down:* DPO trained on noisy labels will memorise the noise rather than learning genuine preferences. See the IPO variant for a mitigation.

**Assumption 4: The partition function cancelling is valid.**

The partition function $Z(x)$ cancels in the pairwise preference probability. This cancellation requires that $Z(x)$ does not depend on $y$ — it is a function of $x$ only. This is correct by definition, so this assumption is always valid. However, the cancellation does mean that DPO never has access to information about the absolute scale of the reward — only the relative ranking of pairs.

**Summary table:**

| Assumption | Condition for validity | Common violation |
|---|---|---|
| On-policy data | Responses sampled from reference policy | Off-policy or human-written responses |
| Bradley-Terry | Scalar, consistent preferences | Multidimensional, noisy human preferences |
| Noise-free labels | Annotator agreement > 95% | Real annotation data (~65–75% agreement) |
| Single reference | Fixed $\pi_\text{ref}$ throughout training | Iterative DPO changes the effective reference |
