# Worked Problem 01: RLHF Objective Derivation

**Topic:** Deriving the RLHF objective from reward maximisation with a KL constraint through to the PPO training loss.

**Difficulty:** Advanced

**Prerequisites:** Basic RL (policy gradient), KL divergence, constrained optimisation (Lagrangians), log-sum-exp.

---

## Problem Statement

Starting from the high-level goal of "train a language model to maximise human reward while staying close to a reference model," derive:

1. The KL-constrained RLHF objective.
2. The closed-form optimal policy under this objective.
3. The PPO surrogate loss used in practice to optimise the policy.
4. The token-level decomposition of the KL penalty.

Show all derivation steps with explanation.

---

## Solution

### Part 1: The KL-Constrained RLHF Objective

Let:
- $x$ denote a prompt sampled from a prompt distribution $\mathcal{D}$
- $y$ denote a complete response (a sequence of tokens)
- $\pi_\theta(y|x)$ be the trainable language model policy
- $\pi_\text{ref}(y|x)$ be the frozen reference policy (the SFT model)
- $r(x, y)$ be a scalar reward assigned by the reward model

The naive objective — maximise expected reward — is:

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} [r(x, y)]$$

This objective alone is insufficient because without constraints, the policy will exploit the reward model. The reward model is a proxy trained on a finite dataset; it assigns high scores to many out-of-distribution responses that are not genuinely good.

To prevent this, we add a KL divergence penalty that penalises the policy for deviating from the reference policy:

$$\boxed{\max_\theta \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot \mathbb{E}_{x \sim \mathcal{D}} \left[ \text{KL}\!\left[ \pi_\theta(\cdot|x) \,\|\, \pi_\text{ref}(\cdot|x) \right] \right]}$$

Expanding the KL using its definition $\text{KL}[P\|Q] = \sum_y P(y) \log \frac{P(y)}{Q(y)}$:

$$= \max_\theta \mathbb{E}_{x,\, y \sim \pi_\theta} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \right]$$

The scalar $\beta > 0$ controls the strength of the KL constraint. Large $\beta$ keeps the policy close to the reference; small $\beta$ allows more aggressive reward optimisation.

---

### Part 2: Closed-Form Optimal Policy

We solve the optimisation analytically. For a fixed $x$, we seek the distribution $\pi^*(\cdot|x)$ over responses $y$ that maximises:

$$\mathcal{F}[\pi] = \sum_y \pi(y|x) \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} \right]$$

subject to the normalisation constraint $\sum_y \pi(y|x) = 1$.

**Using a Lagrangian**, introduce multiplier $\lambda$:

$$\mathcal{L}(\pi, \lambda) = \sum_y \pi(y|x) \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} \right] + \lambda \left( 1 - \sum_y \pi(y|x) \right)$$

**Taking the functional derivative** with respect to $\pi(y|x)$ and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_\text{ref}(y|x)} - \beta - \lambda = 0$$

Rearranging for $\pi(y|x)$:

$$\log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} = \frac{r(x,y)}{\beta} - \frac{\lambda + \beta}{\beta}$$

$$\pi^*(y|x) = \pi_\text{ref}(y|x) \cdot \exp\!\left(\frac{r(x,y)}{\beta}\right) \cdot \exp\!\left(-\frac{\lambda + \beta}{\beta}\right)$$

The second exponential is a constant with respect to $y$ — it is determined by the normalisation constraint:

$$\sum_y \pi^*(y|x) = 1 \implies \exp\!\left(-\frac{\lambda + \beta}{\beta}\right) = \frac{1}{Z(x)}$$

where $Z(x) = \sum_y \pi_\text{ref}(y|x) \exp(r(x,y)/\beta)$ is the **partition function**.

Therefore the optimal policy is:

$$\boxed{\pi^*(y|x) = \frac{\pi_\text{ref}(y|x) \cdot \exp\!\left(r(x,y)/\beta\right)}{Z(x)}}$$

**Interpretation:** The optimal policy is the reference policy "tilted" by $\exp(r/\beta)$. High-reward responses are upweighted; low-reward responses are downweighted. The temperature $\beta$ controls how sharply the tilting concentrates mass on the best responses.

Note: $Z(x)$ is intractable to compute exactly (it requires summing over all possible response sequences). This is why we cannot directly sample from $\pi^*$ — we must approximate it via RL.

---

### Part 3: PPO Surrogate Loss

Since $\pi^*$ is intractable, we use Proximal Policy Optimisation (PPO) to approximate the optimum. PPO is an on-policy actor-critic method that makes multiple gradient steps per batch of data while staying close to the behaviour policy that generated the data.

**The RLHF objective restated as a maximisation:**

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \right]$$

In the RL framing, the generation of a response token-by-token is a Markov Decision Process:
- **State** at step $t$: $(x, y_{<t})$ — the prompt and all tokens generated so far.
- **Action** at step $t$: $y_t$ — the next token.
- **Reward:** Sparse — only received at the end of the sequence. Typically the terminal reward is $r(x, y) - \beta \text{KL}_\text{total}$ and intermediate rewards are 0 (or $-\beta \log \frac{\pi_\theta(y_t|y_{<t},x)}{\pi_\text{ref}(y_t|y_{<t},x)}$ per token; see Part 4).

**Advantage estimation.** PPO uses a value function $V_\psi(s_t)$ that estimates the expected future return from state $s_t$. The advantage at time $t$ is:

$$A_t = Q(s_t, a_t) - V_\psi(s_t)$$

estimated using Generalised Advantage Estimation (GAE):

$$\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}, \quad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$

**PPO clipped objective.** To prevent large policy updates that destabilise training, PPO clips the probability ratio:

$$\rho_t(\theta) = \frac{\pi_\theta(y_t | y_{<t}, x)}{\pi_{\theta_\text{old}}(y_t | y_{<t}, x)}$$

The clipped surrogate loss (maximised over $\theta$) is:

$$\mathcal{L}_\text{CLIP}(\theta) = \mathbb{E}_t \left[ \min\!\left( \rho_t(\theta) \hat{A}_t,\; \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

The clip prevents the ratio from moving too far from 1 in either direction. With $\epsilon = 0.2$ (typical), ratio values outside $[0.8, 1.2]$ have their gradient zeroed.

**Full PPO loss.** The complete training loss combines three terms:

$$\mathcal{L}_\text{PPO}(\theta, \psi) = -\mathcal{L}_\text{CLIP}(\theta) + c_V \mathcal{L}_V(\psi) - c_H \mathcal{H}[\pi_\theta]$$

where:
- $\mathcal{L}_V(\psi) = \mathbb{E}_t \left[ (V_\psi(s_t) - R_t)^2 \right]$ is the value function MSE loss.
- $\mathcal{H}[\pi_\theta] = -\mathbb{E}_t [\log \pi_\theta(y_t|s_t)]$ is an entropy bonus that encourages exploration.
- $c_V \approx 0.5$ and $c_H \approx 0.01$ are coefficients.

---

### Part 4: Token-Level Decomposition of the KL Penalty

A response $y = (y_1, y_2, \ldots, y_T)$ is a sequence of tokens. The sequence-level KL divergence decomposes into a sum of token-level KL terms.

Using the chain rule of probability:

$$\pi_\theta(y|x) = \prod_{t=1}^T \pi_\theta(y_t | y_{<t}, x)$$

Therefore:

$$\log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} = \sum_{t=1}^T \log \frac{\pi_\theta(y_t | y_{<t}, x)}{\pi_\text{ref}(y_t | y_{<t}, x)}$$

This sum is the **sequence-level KL divergence** (evaluated on a single sample $y$). In expectation over $y \sim \pi_\theta$, this equals $\text{KL}[\pi_\theta(\cdot|x) \| \pi_\text{ref}(\cdot|x)]$ by definition.

**Implication for implementation:** the KL penalty can be applied per-token as a per-step reward:

$$\tilde{r}_t = \begin{cases} r(x,y) - \beta \log \frac{\pi_\theta(y_T|y_{<T},x)}{\pi_\text{ref}(y_T|y_{<T},x)} & t = T \text{ (final token)} \\ - \beta \log \frac{\pi_\theta(y_t|y_{<t},x)}{\pi_\text{ref}(y_t|y_{<t},x)} & t < T \end{cases}$$

This per-token credit assignment is crucial for the value function: a value function trained on a sparse terminal reward alone struggles to provide useful gradient signal at early token positions. The per-token KL provides a dense reward signal that makes value learning tractable.

---

### Summary of the Derivation

```
Goal: maximise reward while staying close to SFT model
         |
         v
Objective: E[r(x,y)] - beta * KL[pi_theta || pi_ref]
         |
         v (solve via Lagrangian)
Optimal policy: pi*(y|x) = pi_ref(y|x) * exp(r/beta) / Z(x)
         |
         v (intractable -- approximate via RL)
PPO surrogate: clipped ratio loss + value loss + KL penalty
         |
         v (token-level decomposition of KL)
Per-token reward: sparse terminal reward + dense per-token KL penalty
```

---

### Common Mistakes in Interviews

1. **Forgetting the partition function.** The optimal policy $\pi^*(y|x) \propto \pi_\text{ref}(y|x) \exp(r/\beta)$ is not normalised by the naive exponential alone. $Z(x)$ is necessary and non-trivial.

2. **Confusing KL direction.** The RLHF objective uses $\text{KL}[\pi_\theta \| \pi_\text{ref}]$ (forward KL), not $\text{KL}[\pi_\text{ref} \| \pi_\theta]$ (reverse KL). Forward KL penalises the policy for assigning probability to regions where the reference assigns zero probability. Reverse KL would have very different behaviour (mode-seeking vs mean-seeking).

3. **Treating the KL as a post-hoc add-on.** The KL penalty is not an afterthought — it is integral to the problem formulation and arises from the Lagrangian of the constrained optimisation. Without it, the optimal policy is a degenerate delta function on the highest-reward response.

4. **Missing the clipping in PPO.** The PPO loss clips the probability ratio; the plain policy gradient loss does not. The clipping is what makes PPO stable — without it you are running vanilla policy gradient with its high-variance, instability.
