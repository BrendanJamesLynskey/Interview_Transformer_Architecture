# Problem 02: MoE Routing

**Topic:** Mixture of Experts — routing, load balancing loss, and capacity factor

**Difficulty:** Intermediate to Advanced

**Expected time:** 25–35 minutes

---

## Problem Statement

You are working with a Mixture of Experts layer with $E = 6$ experts and top-$k = 2$ routing. A batch contains $T = 12$ tokens.

**Part A.** Given the following router logits for 4 representative tokens, compute the softmax scores, apply top-2 selection with renormalisation, and determine which experts process each token.

Router logits $Z \in \mathbb{R}^{4 \times 6}$ (rows = tokens, columns = experts):

$$Z = \begin{bmatrix}
2.1 & 0.3 & -0.5 & 1.8 & 0.7 & -0.2 \\
-0.4 & 3.2 & 1.1 & 0.2 & 0.8 & 2.9 \\
1.5 & 1.6 & 1.4 & 0.3 & 0.9 & 0.8 \\
0.1 & 0.2 & 0.3 & 2.5 & 2.6 & 0.4
\end{bmatrix}$$

**Part B.** Given the complete routing assignments for all $T = 12$ tokens in the batch (provided below), compute the auxiliary load balancing loss $\mathcal{L}_{\text{aux}}$.

Full batch routing summary (illustrative values for all 12 tokens; counts are top-2 assignments):
- Expert 1: receives 5 assignments with average soft probability $\bar{p}_1 = 0.26$
- Expert 2: receives 4 assignments with average soft probability $\bar{p}_2 = 0.18$
- Expert 3: receives 1 assignment with average soft probability $\bar{p}_3 = 0.06$
- Expert 4: receives 3 assignments with average soft probability $\bar{p}_4 = 0.13$
- Expert 5: receives 7 assignments with average soft probability $\bar{p}_5 = 0.23$
- Expert 6: receives 4 assignments with average soft probability $\bar{p}_6 = 0.14$

Note: with top-2 routing and 12 tokens, the total "slots" is $12 \times 2 = 24$ assignments.

**Part C.** The capacity factor is $C = 1.5$. What is the token capacity per expert? Which experts would overflow, and which tokens (from Part A) get dropped?

**Part D.** Suppose at the next training step the router produces exactly uniform routing (all experts receive the same fraction of tokens and have the same soft probability). What is $\mathcal{L}_{\text{aux}}$ in this case? Why is this the global minimum of the loss?

**Part E.** What are the practical consequences of setting $\alpha$ (the weight of $\mathcal{L}_{\text{aux}}$) too large vs too small?

---

## Solution

### Part A: Softmax scores, top-2 selection, renormalisation

**Token 1 logits:** $[2.1, 0.3, -0.5, 1.8, 0.7, -0.2]$

Softmax computation. First, compute the exponentials (subtracting max $= 2.1$ for stability):

| Expert | Logit | $z - \max$ | $e^{z-\max}$ |
|---|---|---|---|
| 1 | 2.1 | 0.0 | 1.0000 |
| 2 | 0.3 | -1.8 | 0.1653 |
| 3 | -0.5 | -2.6 | 0.0743 |
| 4 | 1.8 | -0.3 | 0.7408 |
| 5 | 0.7 | -1.4 | 0.2466 |
| 6 | -0.2 | -2.3 | 0.1003 |

Sum $= 2.3273$

Softmax scores: $\mathbf{s}_1 = [0.430, 0.071, 0.032, 0.318, 0.106, 0.043]$

**Top-2 selection:** Experts 1 and 4 (scores $0.430$ and $0.318$)

**Renormalised weights:**
$$\tilde{s}_1 = \frac{0.430}{0.430 + 0.318} = \frac{0.430}{0.748} = 0.575, \quad \tilde{s}_4 = \frac{0.318}{0.748} = 0.425$$

**Token 1 output:** $y_1 = 0.575 \cdot f_1(x_1) + 0.425 \cdot f_4(x_1)$

---

**Token 2 logits:** $[-0.4, 3.2, 1.1, 0.2, 0.8, 2.9]$

Max $= 3.2$. Shifted: $[-3.6, 0.0, -2.1, -3.0, -2.4, -0.3]$

| Expert | $e^{z-\max}$ | Softmax |
|---|---|---|
| 1 | 0.0273 | 0.013 |
| 2 | 1.0000 | 0.492 |
| 3 | 0.1225 | 0.060 |
| 4 | 0.0498 | 0.025 |
| 5 | 0.0907 | 0.045 |
| 6 | 0.7408 | 0.365 |

Sum: $0.0273 + 1.0000 + 0.1225 + 0.0498 + 0.0907 + 0.7408 = 2.0311$

Softmax: $[0.013, 0.492, 0.060, 0.025, 0.045, 0.365]$

**Top-2:** Experts 2 and 6 (scores $0.492$ and $0.365$)

Renormalised: $\tilde{s}_2 = 0.492/0.857 = 0.574$, $\tilde{s}_6 = 0.365/0.857 = 0.426$

---

**Token 3 logits:** $[1.5, 1.6, 1.4, 0.3, 0.9, 0.8]$

Max $= 1.6$. Shifted: $[-0.1, 0.0, -0.2, -1.3, -0.7, -0.8]$

Exponentials: $[0.9048, 1.0000, 0.8187, 0.2725, 0.4966, 0.4493]$

Sum $= 3.9419$

Softmax: $[0.2294, 0.2537, 0.2077, 0.0691, 0.1259, 0.1140]$

**Top-2:** Experts 2 and 1 (scores $0.2537$ and $0.2294$) — very close!

Renormalised: $\tilde{s}_2 = 0.2537/0.4831 = 0.525$, $\tilde{s}_1 = 0.2294/0.4831 = 0.475$

**Note:** Token 3 is nearly indifferent between experts 1 and 2. This is a "near-tie" case where a small perturbation could change the routing decision. In practice, this ambiguity is fine — either expert produces a reasonable output, and training adjusts accordingly.

---

**Token 4 logits:** $[0.1, 0.2, 0.3, 2.5, 2.6, 0.4]$

Max $= 2.6$. Shifted: $[-2.5, -2.4, -2.3, -0.1, 0.0, -2.2]$

Exponentials: $[0.0821, 0.0907, 0.1003, 0.9048, 1.0000, 0.1108]$

Sum $= 2.2887$

Softmax: $[0.036, 0.040, 0.044, 0.395, 0.437, 0.048]$

**Top-2:** Experts 5 and 4 (scores $0.437$ and $0.395$)

Renormalised: $\tilde{s}_5 = 0.437/0.832 = 0.525$, $\tilde{s}_4 = 0.395/0.832 = 0.475$

---

**Summary of Part A routing:**

| Token | Expert 1 | Expert 2 | Weights |
|---|---|---|---|
| 1 | Expert 1 | Expert 4 | 0.575, 0.425 |
| 2 | Expert 2 | Expert 6 | 0.574, 0.426 |
| 3 | Expert 2 | Expert 1 | 0.525, 0.475 |
| 4 | Expert 5 | Expert 4 | 0.525, 0.475 |

---

### Part B: Auxiliary load balancing loss

**Setup.** $E = 6$ experts, $T = 12$ tokens, top-$k = 2$, so total assignments $= 24$.

The auxiliary loss formula (Switch Transformer style):

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{e=1}^E f_e \cdot p_e$$

where $f_e$ = fraction of assignments going to expert $e$, and $p_e$ = mean soft router probability for expert $e$.

**Consistency checks.** Each of the 12 tokens is sent to exactly 2 experts, so the assignment counts must sum to $T \times k = 24$: $5 + 4 + 1 + 3 + 7 + 4 = 24$. ✓ Each token's softmax sums to 1, so the mean soft probabilities must also sum to 1: $0.26 + 0.18 + 0.06 + 0.13 + 0.23 + 0.14 = 1.00$. ✓

**Compute $f_e$** (assignments received / total assignments $= \text{count}_e / 24$) and the products $f_e \cdot p_e$:

| Expert | $\text{count}_e$ | $f_e = \text{count}_e/24$ | $p_e$ | $f_e \cdot p_e$ |
|---|---|---|---|---|
| 1 | 5 | 0.2083 | 0.26 | 0.0542 |
| 2 | 4 | 0.1667 | 0.18 | 0.0300 |
| 3 | 1 | 0.0417 | 0.06 | 0.0025 |
| 4 | 3 | 0.1250 | 0.13 | 0.0163 |
| 5 | 7 | 0.2917 | 0.23 | 0.0671 |
| 6 | 4 | 0.1667 | 0.14 | 0.0233 |

Sum: $\sum_e f_e \cdot p_e = \frac{5(0.26) + 4(0.18) + 1(0.06) + 3(0.13) + 7(0.23) + 4(0.14)}{24} = \frac{4.64}{24} = 0.1933$

$$\mathcal{L}_{\text{aux}} = \alpha \times 6 \times 0.1933 = 1.16\alpha$$

For typical $\alpha = 0.01$: $\mathcal{L}_{\text{aux}} = 0.0116$

Compare with the balanced minimum of $1.00\alpha$ (Part D): this batch is 16% above the minimum.

**Interpretation.** The largest contributors to the loss are experts 5 and 1, which are overloaded (7 and 5 assignments against a balanced 4). Expert 3 is severely underloaded (only 1 assignment, low soft probability). The loss gradient with respect to $p_e$ is proportional to $f_e$, so it pushes hardest on the soft probabilities of experts 5 and 1 and least on expert 3.

---

### Part C: Capacity factor and token dropping

**Capacity per expert:**

$$\text{capacity} = C \times \frac{T \times k}{E} = 1.5 \times \frac{12 \times 2}{6} = 1.5 \times 4 = 6 \text{ tokens per expert}$$

**Which experts overflow?**

| Expert | Assignments | Capacity | Overflow? |
|---|---|---|---|
| 1 | 5 | 6 | No |
| 2 | 4 | 6 | No |
| 3 | 1 | 6 | No |
| 4 | 3 | 6 | No |
| 5 | 7 | 6 | **Yes, by 1** |
| 6 | 4 | 6 | No |

With $C = 1.5$, Expert 5 overflows by one assignment. The assignment with the lowest softmax score for Expert 5 is dropped (priority ordering: tokens are processed in order of decreasing softmax score for that expert; an overflowing assignment is skipped, and if a token loses all its experts its output equals its input $x$ via the residual connection). Note that Expert 3 has 5 unused slots at the same time — capacity is wasted and tokens are dropped in the same batch, which is exactly what the auxiliary loss is meant to prevent.

**From Part A.** Token 4 routes to Expert 5 with weight $0.525$ and Expert 4 with weight $0.475$. If Token 4's Expert 5 score ($0.437$) is the lowest among the 7 tokens routed to Expert 5, its assignment to Expert 5 is the one dropped. The output would then be:

$$y_4 = 0 + f_4(x_4) \quad \text{(only Expert 4 contribution, unweighted)}$$

or alternatively:

$$y_4 = x_4 + f_4(x_4) \quad \text{(residual pass-through + Expert 4)}$$

depending on the implementation. The quality impact is that Token 4 receives only one expert's processing instead of two, effectively degrading the model's expressivity for that token.

---

### Part D: Uniform routing — global minimum of $\mathcal{L}_{\text{aux}}$

**Perfect balance setup:** All experts receive $T/E = 12/6 = 2$ tokens (with top-2, $f_e = 2 \times 2 / (12 \times 2) = 4/24 = 1/6$ for all $e$). All soft probabilities equal $p_e = 1/E = 1/6$ for all $e$.

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{e=1}^E f_e \cdot p_e = \alpha \cdot 6 \sum_{e=1}^6 \frac{1}{6} \cdot \frac{1}{6} = \alpha \cdot 6 \cdot 6 \cdot \frac{1}{36} = \alpha$$

So at perfect balance, $\mathcal{L}_{\text{aux}} = \alpha$.

**Why this is the global minimum.** For arbitrary distributions $f$ and $p$, $\sum_e f_e p_e$ could be made smaller by anti-aligning them (large $f_e$ paired with small $p_e$). But the router cannot do that: $f$ is the hard (top-$k$) version of $p$, so experts with high soft probability are the ones that receive tokens, and the two are aligned. In the idealised case $f = p$:

$$E \sum_e p_e^2 \geq E \cdot \frac{\left(\sum_e p_e\right)^2}{E} = 1$$

by the Cauchy–Schwarz inequality, with equality if and only if $p_e = 1/E$ for all $e$. So the loss is minimised, at $\alpha \cdot E \cdot 1/E = \alpha$, by uniform routing — the Switch Transformer paper's rationale for this loss.

---

### Part E: Consequences of $\alpha$ too large vs too small

**$\alpha$ too small (e.g., $\alpha = 10^{-4}$):**
- The main language modelling loss $\mathcal{L}_{\text{LM}} \sim 1\text{–}3$ dominates; $\mathcal{L}_{\text{aux}} \sim 10^{-4}$ is negligible
- The router collapses: 1–2 popular experts receive most tokens
- Under-utilised experts receive few gradient updates and atrophy
- The model degenerates toward a small dense model with wasted parameters
- This is the most common failure mode in early MoE training without careful tuning

**$\alpha$ too large (e.g., $\alpha = 1.0$):**
- The load balancing loss dominates training
- The router is forced toward uniform routing regardless of input content
- Expert specialisation is suppressed — all experts learn similar functions
- The model loses the benefit of expert diversity
- Perplexity degrades: the model becomes equivalent to a single expert (uniform averaging of all experts' outputs)

**Recommended range.** $\alpha \in [0.01, 0.1]$ for most MoE models. Switch Transformer uses $\alpha = 10^{-2}$; Mixtral reports no auxiliary loss (relying on training dynamics). The correct value is empirical and depends on the number of experts, model size, and task.

**Alternative: $z$-loss.** Some implementations add a "router $z$-loss" that penalises large logit magnitudes:
$$\mathcal{L}_z = \beta \cdot \frac{1}{T} \sum_t \left(\log \sum_e e^{z_{t,e}}\right)^2$$

This prevents the router from producing extremely sharp distributions (numerical instability) while being softer than load balancing. Used in ST-MoE (Zoph et al., 2022).

---

## Implementation Reference

```python
import torch
import torch.nn.functional as F

def top_k_routing(logits: torch.Tensor, k: int = 2):
    """
    Compute top-k MoE routing.

    Args:
        logits: (T, E) router logits for T tokens and E experts
        k: number of active experts per token

    Returns:
        weights: (T, k) renormalised routing weights
        indices: (T, k) selected expert indices
    """
    T, E = logits.shape

    # Softmax scores
    scores = F.softmax(logits, dim=-1)  # (T, E)

    # Top-k selection
    top_scores, top_indices = torch.topk(scores, k, dim=-1)  # (T, k) each

    # Renormalise so selected weights sum to 1
    top_weights = top_scores / top_scores.sum(dim=-1, keepdim=True)  # (T, k)

    return top_weights, top_indices


def auxiliary_load_balancing_loss(logits: torch.Tensor, indices: torch.Tensor, alpha: float = 0.01):
    """
    Compute Switch Transformer auxiliary load balancing loss.

    Args:
        logits: (T, E) raw router logits
        indices: (T, k) selected expert indices (from top-k routing)
        alpha: loss coefficient

    Returns:
        scalar loss
    """
    T, E = logits.shape
    k = indices.shape[1]

    # Soft probabilities p_e: mean softmax probability for each expert
    scores = F.softmax(logits, dim=-1)  # (T, E)
    p = scores.mean(dim=0)  # (E,) -- average soft probability per expert

    # Hard routing fractions f_e: fraction of total assignments going to each expert
    # One-hot encode the selected experts
    one_hot = torch.zeros(T, E, device=logits.device)
    for ki in range(k):
        one_hot.scatter_(1, indices[:, ki:ki+1], 1.0)
    # f_e = total assignments to expert e / (T * k)
    f = one_hot.sum(dim=0) / (T * k)  # (E,)

    # Auxiliary loss: E * sum(f_e * p_e)
    loss = alpha * E * (f * p).sum()
    return loss


# Test with Part A logits
Z = torch.tensor([
    [2.1, 0.3, -0.5, 1.8, 0.7, -0.2],
    [-0.4, 3.2, 1.1, 0.2, 0.8, 2.9],
    [1.5, 1.6, 1.4, 0.3, 0.9, 0.8],
    [0.1, 0.2, 0.3, 2.5, 2.6, 0.4],
])

weights, indices = top_k_routing(Z, k=2)
print("Routing weights (top-2):")
for t in range(4):
    exp_pair = indices[t].tolist()
    w_pair = weights[t].tolist()
    print(f"  Token {t+1}: Expert {exp_pair[0]+1} ({w_pair[0]:.3f}), Expert {exp_pair[1]+1} ({w_pair[1]:.3f})")

# Compute load balancing loss
loss = auxiliary_load_balancing_loss(Z, indices, alpha=0.01)
print(f"\nAuxiliary loss (alpha=0.01): {loss.item():.6f}")
```

**Expected output:**
```
Routing weights (top-2):
  Token 1: Expert 1 (0.574), Expert 4 (0.426)
  Token 2: Expert 2 (0.574), Expert 6 (0.426)
  Token 3: Expert 2 (0.525), Expert 1 (0.475)
  Token 4: Expert 5 (0.525), Expert 4 (0.475)

Auxiliary loss (alpha=0.01): 0.011304
```
