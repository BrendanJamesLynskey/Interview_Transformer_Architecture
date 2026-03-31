# DPO: Direct Preference Optimisation — Interview Questions

DPO (Rafailov et al., 2023) is a preference learning algorithm that eliminates the explicit reward model and RL loop of RLHF by directly fine-tuning the language model on preference data. It has become one of the most widely used alignment methods due to its simplicity and stability. Questions on DPO are now standard in ML research and engineering interviews.

---

## Fundamentals

### Q1. What problem does DPO solve, and what is the key insight?

**Answer.**

RLHF requires training a separate reward model and then running PPO, which involves four models simultaneously (policy, reference, reward, value function), complex hyperparameter tuning, and unstable training dynamics.

DPO's key insight is that the RLHF objective has a closed-form optimal policy (derived in RLHF Q8):

$$\pi^*(y|x) = \frac{\pi_\text{ref}(y|x) \exp(r(x,y)/\beta)}{Z(x)}$$

This can be rearranged to express the reward in terms of the optimal policy:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

The key insight: **the reward function is implicitly defined by the policy**. We do not need to train a separate reward model — we can substitute this implicit reward into the preference learning objective and directly optimise the policy.

The partition function $Z(x)$ cancels when computing pairwise preference probabilities, yielding a stable objective that can be optimised with standard supervised learning, bypassing RL entirely.

---

### Q2. State the DPO loss and explain each term.

**Answer.**

The DPO loss is:

$$\mathcal{L}_\text{DPO}(\pi_\theta; \pi_\text{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

Breaking down the terms:

- $y_w$: the preferred ("winner") response in the comparison pair.
- $y_l$: the dispreferred ("loser") response.
- $\pi_\theta(y|x)$: the log-probability of generating response $y$ given prompt $x$ under the current (trainable) policy.
- $\pi_\text{ref}(y|x)$: the same log-probability under the frozen reference model (the SFT initialisation).
- $\log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$: the **implicit reward** — how much more (or less) likely the trainable policy makes response $y$ compared to the reference, on a log scale. This is the change in relative probability induced by fine-tuning.
- $\beta$: temperature controlling how strongly the policy is allowed to deviate from the reference.
- $\sigma(\cdot)$: the sigmoid function.

The loss pushes $\pi_\theta$ to increase the implicit reward of $y_w$ relative to $y_l$: it encourages the policy to increase the probability of the preferred response and decrease the probability of the dispreferred response, while the KL constraint (implicit via the log ratio terms) prevents over-deviation from the reference.

---

### Q3. What training data does DPO require?

**Answer.**

DPO requires a dataset of **preference triplets**:

$$\mathcal{D} = \{(x^{(i)},\, y_w^{(i)},\, y_l^{(i)})\}_{i=1}^N$$

Each triplet contains:
- A prompt $x$
- A preferred response $y_w$ (the human-chosen "better" response)
- A dispreferred response $y_l$ (the human-chosen "worse" response)

This is exactly the same data format used to train a reward model in RLHF. In fact, DPO can be applied directly to existing RLHF preference datasets such as Anthropic's HH-RLHF or the OpenAI WebGPT comparisons dataset.

**Important subtlety: on-policy vs off-policy data.**

The DPO derivation assumes the response pairs $(y_w, y_l)$ were sampled from the reference policy $\pi_\text{ref}$. In practice, preference datasets are often collected from a different (typically stronger) model or from human demonstrations. When the preference data is far from the reference policy's distribution, DPO can degrade because the log-ratio terms become inaccurate estimates of the implicit reward. This is known as the **off-policy distribution mismatch** problem and is the main motivation for iterative DPO variants that regenerate preference data from the current policy.

---

### Q4. Qualitatively, how does the DPO gradient update the policy?

**Answer.**

The gradient of the DPO loss with respect to the policy parameters pushes in two directions simultaneously:

1. **Increase the log-probability of the preferred response $y_w$** under $\pi_\theta$, weighted by how "surprised" the model is (i.e., how much the preferred response is currently under-ranked relative to the reference).

2. **Decrease the log-probability of the dispreferred response $y_l$** under $\pi_\theta$, weighted by the same surprise factor.

The weighting by the sigmoid term $\sigma(\hat{r}_l - \hat{r}_w)$ (where $\hat{r}$ denotes the implicit reward) is important: pairs where the model already correctly ranks $y_w > y_l$ contribute little gradient. Pairs where the model incorrectly ranks $y_l > y_w$ contribute large gradients. This makes DPO focus learning on its current mistakes, similar to how hard negative mining works in metric learning.

---

## Intermediate

### Q5. Derive the DPO loss from the RLHF objective.

**Answer.**

**Step 1: Recall the RLHF objective and its optimal policy.**

The KL-constrained RLHF objective:
$$\max_\pi \mathbb{E}_{y \sim \pi} [r(x,y)] - \beta \, \text{KL}[\pi \| \pi_\text{ref}]$$

has optimal solution:
$$\pi^*(y|x) = \frac{\pi_\text{ref}(y|x) \exp(r(x,y)/\beta)}{Z(x)}$$

**Step 2: Rearrange for the reward.**

Taking the log:
$$\log \pi^*(y|x) = \log \pi_\text{ref}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)$$

Rearranging:
$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$$

**Step 3: Substitute into the Bradley-Terry preference model.**

The probability of preferring $y_w$ over $y_l$ under the Bradley-Terry model is:
$$P(y_w \succ y_l | x) = \sigma(r(x,y_w) - r(x,y_l))$$

Substituting the reward expression:
$$r(x,y_w) - r(x,y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)} + \underbrace{\beta \log Z(x) - \beta \log Z(x)}_{=0}$$

The partition function $Z(x)$ cancels entirely, giving:
$$P(y_w \succ y_l | x) = \sigma\!\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)} \right)$$

**Step 4: Write the maximum likelihood loss.**

Replacing the unknown optimal policy $\pi^*$ with the trainable policy $\pi_\theta$ and taking the negative log-likelihood over the dataset:

$$\boxed{\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_\mathcal{D} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]}$$

This is the DPO loss. Notice that no reward model and no RL are needed — the partition function cancelled, leaving a purely supervised loss.

---

### Q6. Compare DPO and RLHF on computational requirements, stability, and performance.

**Answer.**

| Dimension | RLHF (PPO) | DPO |
|---|---|---|
| Models in memory | 4 (policy, reference, reward, value) | 2 (policy, reference) |
| Training paradigm | RL (online, on-policy) | Supervised (offline, off-policy) |
| Stability | Requires careful tuning of many hyperparameters | Stable; behaves like SFT |
| Reward model | Explicit, separately trained | Implicit, no separate model |
| Compute per step | 4–5x SFT step | ~2x SFT step |
| On-policy data requirement | Yes — generates responses during training | No — trains on static preference pairs |
| Exploration | Policy actively explores the response space | No exploration; distribution fixed at data collection |
| Performance ceiling | Higher; can discover novel strategies | Lower; limited by training data distribution |

**When to prefer each:**

DPO is preferred when:
- Computational resources are limited
- A high-quality off-policy preference dataset already exists
- Training stability is critical (production pipelines, reproducibility requirements)
- The model is relatively close to the reference model (small policy-reference gap)

RLHF/PPO is preferred when:
- Maximising alignment quality is the priority
- The task requires the model to explore novel response strategies (e.g., multi-step reasoning, tool use)
- You can afford to iterate on online preference data
- The task has a verifiable reward (e.g., coding, math) rather than just human preference

In practice, many teams use DPO as a first step and add PPO only when DPO performance plateaus.

---

### Q7. Explain the IPO and KTO variants of DPO. What problems do they address?

**Answer.**

**IPO (Identity Preference Optimisation, Azar et al., 2023)**

IPO addresses the **overfitting problem** in DPO. The DPO loss uses a sigmoid of the difference in implicit rewards, which saturates when the model strongly prefers $y_w$ over $y_l$. In the limit, the policy can achieve zero training loss by assigning zero probability to $y_l$ and near-certain probability to $y_w$. This causes the policy to overfit to the preference labels and can degenerate (making responses in $y_l$ have $-\infty$ log-probability).

IPO replaces the sigmoid with a squared loss that does not saturate:

$$\mathcal{L}_\text{IPO}(\pi_\theta) = \mathbb{E}_\mathcal{D} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]$$

This targets an implicit reward difference of $\frac{1}{2\beta}$ rather than infinity, preventing saturation.

**KTO (Kahneman-Tversky Optimisation, Ethayarajh et al., 2024)**

KTO addresses the **paired data requirement** of DPO. DPO requires paired comparisons $(y_w, y_l)$ for the same prompt. But in many settings you have unpaired data: a collection of (prompt, response, label) triples where each response is simply labelled as "good" or "bad," not compared to another response.

KTO is motivated by Kahneman and Tversky's prospect theory, which models human utility asymmetrically (losses are felt more strongly than gains). The KTO loss treats chosen and rejected responses separately:

$$\mathcal{L}_\text{KTO}(\pi_\theta) = \mathbb{E}_\mathcal{D} \left[ w(y) \cdot \left( 1 - \sigma\!\left( \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} - z_\text{ref} \right) \right) \right]$$

where $z_\text{ref}$ is a reference point (the expected KL of the current policy from the reference), and $w(y)$ is positive for chosen responses and negative for rejected responses. This allows training on unpaired binary feedback, which is much cheaper to collect.

---

## Advanced

### Q8. What is the off-policy distribution mismatch problem in DPO, and how do iterative DPO methods address it?

**Answer.**

**The problem.**

DPO is derived under the assumption that the preference data $(y_w, y_l)$ was sampled from the reference policy $\pi_\text{ref}$. The implicit reward $\beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ only approximates the true reward accurately in regions of response space where $\pi_\text{ref}$ has significant probability.

In practice, preference datasets are often collected from humans or from a different (often stronger) model. The responses may be far from the reference policy's distribution. When $y_w$ or $y_l$ are highly improbable under $\pi_\text{ref}$, the log-ratio terms become large and noisy, and the DPO gradient becomes uninformative or misleading.

Empirically, this manifests as:
- The policy memorises the training pairs rather than learning general preference structure.
- The policy's probability of preferred responses can actually *decrease* during training (the "probability degradation" problem documented in several papers).
- Poor generalisation to prompts not represented in the preference dataset.

**Iterative DPO (also called Online DPO or Self-Play DPO).**

The solution is to periodically re-generate response pairs from the *current* policy and re-collect or re-infer preferences:

1. Start with the reference policy $\pi_\text{ref}^{(0)}$ (SFT model).
2. Train DPO on an existing preference dataset to get $\pi_\theta^{(1)}$.
3. Sample $(y_w, y_l)$ from $\pi_\theta^{(1)}$ for each prompt. Rank with a reward model or human annotators.
4. Train DPO on this new on-policy dataset to get $\pi_\theta^{(2)}$.
5. Repeat.

At each iteration, the preference data is on-policy for the current model, reducing the distribution mismatch. This is analogous to how PPO is inherently on-policy. SPIN (Self-Play Fine-Tuning) and RLCD are variants of this approach.

---

### Q9. How does DPO relate to contrastive learning, and what is the "implicit reward" interpretation?

**Answer.**

The DPO loss can be viewed as a contrastive objective over the space of responses. The log-ratio $\log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ plays the role of a scoring function that measures how much the trainable policy has moved the probability of response $y$ relative to the reference. DPO trains this implicit reward to rank $y_w$ above $y_l$.

This is structurally analogous to metric learning with pairwise contrastive loss: the "anchor" is the prompt $x$, the "positive" is $y_w$, and the "negative" is $y_l$. The embedding is the log-ratio (implicit reward).

**Implicit reward interpretation.**

Define the implicit reward as:
$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$$

After DPO training, $\hat{r}_\theta$ can be used as a reward model without training a separate classifier. Rafailov et al. showed that DPO-trained models produce implicit reward orderings that correlate well with human preferences, validating this interpretation.

This has practical implications:
- DPO-trained models can be used for reward labelling in subsequent preference data collection (self-rewarding).
- The implicit reward can be used for best-of-N reranking at inference time.
- The log-ratio gap $\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$ measured on the training set can be used as a data quality proxy: pairs with a large gap are "easy" for the model; pairs with a small gap (or negative gap) are "hard" and may indicate noisy labels.

---

### Q10. What are the practical failure modes of DPO, and how do you debug them?

**Answer.**

**1. Probability mass collapse on rejected responses.**

In some implementations, the policy assigns near-zero probability to $y_l$ responses. This causes underflow in the log-ratio, numerical instability, and degenerate gradients. The policy may produce `NaN` losses.

*Fix:* add a reference anchor regularisation term (as in IPO), or clip log-ratios to a finite range. Monitor the distribution of $\log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}$ — it should not diverge to $-\infty$.

**2. The policy does not increase $P(y_w)$.**

Documented by Pal et al. (2024) and others: DPO can decrease the probability of both $y_w$ and $y_l$, with the loss minimised by decreasing $P(y_l)$ more than $P(y_w)$. The model technically satisfies the preference ranking but becomes less likely to generate either response.

*Fix:* add an SFT loss on $y_w$ alongside the DPO loss: $\mathcal{L} = \mathcal{L}_\text{DPO} + \alpha \mathcal{L}_\text{SFT}(y_w)$. This anchors the model to the preferred responses.

**3. Off-policy distribution mismatch** (discussed in Q8).

*Fix:* use iterative DPO or filter preference data to pairs that are within the support of the reference policy.

**4. Noisy preference labels.**

Human preference data has label noise: annotators disagree, and some pairs have ambiguous preferences. DPO is sensitive to noise because it treats all preference labels as ground truth.

*Fix:* filter training pairs by annotator agreement rate. Discard pairs with low inter-annotator agreement. Consider confidence-weighted DPO loss where noisier pairs contribute less gradient.

**5. Hyperparameter sensitivity to $\beta$.**

$\beta$ controls how much the policy can deviate from the reference. Too small: the policy over-optimises and reward-hacks the training pairs. Too large: the policy barely changes from the SFT model.

*Fix:* start with $\beta \in [0.1, 0.5]$ and tune by monitoring both the implicit reward gap and downstream eval metrics. The optimal $\beta$ is task-dependent.
