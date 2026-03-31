# Worked Problem 01 — Cross-Entropy Loss for Next-Token Prediction

## Problem Statement

A language model is being trained on next-token prediction. You are given the
following setup:

- Vocabulary size: $V = 6$ (toy example)
- Sequence length being evaluated: $T = 3$ target tokens
- The model has produced raw logit vectors (one per position) **before** the
  target token at each step

**Raw logits** for each prediction step (each row is the logit vector over the
vocabulary at one position):

$$
Z = \begin{bmatrix}
2.1 & 0.3 & -1.2 & 0.8 & 1.5 & -0.4 \\
-0.5 & 3.2 & 0.1 & -1.0 & 0.2 & 1.1 \\
1.0 & 0.5 & 2.8 & 0.3 & -0.7 & 1.4
\end{bmatrix}
$$

Row 1 is the logit vector predicting token at position 2, row 2 predicts position
3, row 3 predicts position 4 (the model always predicts the *next* token).

**Target token indices** (0-indexed): $y = [0,\ 1,\ 2]$

That is:
- Step 1: correct next token is vocabulary item 0 (logit vector row 1)
- Step 2: correct next token is vocabulary item 1 (logit vector row 2)
- Step 3: correct next token is vocabulary item 2 (logit vector row 3)

**Task.** Compute the cross-entropy loss step by step:

1. Compute the softmax probability distribution at each step.
2. Extract the log-probability of the correct token at each step.
3. Compute the per-token loss (negative log-likelihood).
4. Compute the mean loss (the scalar that the optimiser minimises).

---

## Solution

### Step 1 — Softmax at Each Position

The softmax of a logit vector $z \in \mathbb{R}^V$ is:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

For numerical stability, it is standard to subtract the row maximum before
exponentiating:

$$
p_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^{V} e^{z_j - \max_j z_j}}
$$

This does not change the result because the constant cancels in numerator and
denominator.

**Position 1** — logits: $[2.1,\ 0.3,\ -1.2,\ 0.8,\ 1.5,\ -0.4]$, max $= 2.1$

Shifted: $[0.0,\ -1.8,\ -3.3,\ -1.3,\ -0.6,\ -2.5]$

Exponentials:

$$
e^{0.0} = 1.0000,\quad e^{-1.8} = 0.1653,\quad e^{-3.3} = 0.0369
$$
$$
e^{-1.3} = 0.2725,\quad e^{-0.6} = 0.5488,\quad e^{-2.5} = 0.0821
$$

Sum $= 1.0000 + 0.1653 + 0.0369 + 0.2725 + 0.5488 + 0.0821 = 2.1056$

Softmax probabilities:

$$
p^{(1)} = [0.4749,\ 0.0785,\ 0.0175,\ 0.1294,\ 0.2607,\ 0.0390]
$$

**Position 2** — logits: $[-0.5,\ 3.2,\ 0.1,\ -1.0,\ 0.2,\ 1.1]$, max $= 3.2$

Shifted: $[-3.7,\ 0.0,\ -3.1,\ -4.2,\ -3.0,\ -2.1]$

Exponentials:

$$
e^{-3.7} = 0.0247,\quad e^{0.0} = 1.0000,\quad e^{-3.1} = 0.0450
$$
$$
e^{-4.2} = 0.0150,\quad e^{-3.0} = 0.0498,\quad e^{-2.1} = 0.1225
$$

Sum $= 0.0247 + 1.0000 + 0.0450 + 0.0150 + 0.0498 + 0.1225 = 1.2570$

Softmax probabilities:

$$
p^{(2)} = [0.0196,\ 0.7955,\ 0.0358,\ 0.0119,\ 0.0396,\ 0.0975]
$$

**Position 3** — logits: $[1.0,\ 0.5,\ 2.8,\ 0.3,\ -0.7,\ 1.4]$, max $= 2.8$

Shifted: $[-1.8,\ -2.3,\ 0.0,\ -2.5,\ -3.5,\ -1.4]$

Exponentials:

$$
e^{-1.8} = 0.1653,\quad e^{-2.3} = 0.1003,\quad e^{0.0} = 1.0000
$$
$$
e^{-2.5} = 0.0821,\quad e^{-3.5} = 0.0302,\quad e^{-1.4} = 0.2466
$$

Sum $= 0.1653 + 0.1003 + 1.0000 + 0.0821 + 0.0302 + 0.2466 = 1.6245$

Softmax probabilities:

$$
p^{(3)} = [0.1018,\ 0.0617,\ 0.6156,\ 0.0505,\ 0.0186,\ 0.1518]
$$

---

### Step 2 — Log-Probability of the Correct Token

For each position $t$, the cross-entropy loss requires $\log p_{y_t}^{(t)}$ where
$y_t$ is the target token index.

Taking natural log:

| Position | Target index $y_t$ | $p_{y_t}^{(t)}$ | $\log p_{y_t}^{(t)}$ |
|----------|--------------------|------------------|----------------------|
| 1        | 0                  | 0.4749           | $-0.7444$            |
| 2        | 1                  | 0.7955           | $-0.2291$            |
| 3        | 2                  | 0.6156           | $-0.4853$            |

**Efficient computation note.** In practice, we do not compute the softmax
explicitly and then take the log. Instead, we use the **log-sum-exp** identity:

$$
\log p_{y_t} = z_{y_t} - \log\!\sum_{j=1}^{V} e^{z_j}
= z_{y_t} - \log\!\sum_{j=1}^{V} e^{z_j - z_\max} - z_\max
$$

which is numerically stable and avoids computing all probabilities. In PyTorch
this is `F.cross_entropy(logits, targets)` which fuses softmax and log
internally.

Verification for position 1:

$$
\log p_0^{(1)} = 2.1 - \log(e^{2.1} + e^{0.3} + e^{-1.2} + e^{0.8} + e^{1.5} + e^{-0.4})
$$
$$
= 2.1 - \log(8.1662 + 1.3499 + 0.3012 + 2.2255 + 4.4817 + 0.6703)
= 2.1 - \log(17.1948)
= 2.1 - 2.8446 = -0.7446
$$

(Small rounding difference from the two-stage approach above; both are correct.)

---

### Step 3 — Per-Token Negative Log-Likelihood

The per-token loss is the **negative** log-probability:

$$
\ell_t = -\log p_{y_t}^{(t)}
$$

| Position | $-\log p_{y_t}^{(t)}$ | Interpretation |
|----------|-----------------------|----------------|
| 1        | 0.7444                | Model assigned 47.5% to correct token — moderate |
| 2        | 0.2291                | Model assigned 79.6% to correct token — confident and correct |
| 3        | 0.4853                | Model assigned 61.6% to correct token — fairly confident |

---

### Step 4 — Mean Cross-Entropy Loss

The scalar loss used for backpropagation is the **mean** over all $T$ target
positions:

$$
\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \ell_t
= \frac{1}{3}(0.7444 + 0.2291 + 0.4853)
= \frac{1.4588}{3}
= \boxed{0.4863}
$$

---

### Step 5 — Perplexity

Perplexity is the exponentiated average cross-entropy loss and is the standard
reporting metric for language model quality:

$$
\text{PPL} = \exp(\mathcal{L}) = e^{0.4863} \approx 1.627
$$

For a vocabulary of size 6, a random model would achieve $\mathcal{L} = \log 6
\approx 1.792$ and $\text{PPL} = 6$. Our model's perplexity of 1.63 indicates
it is substantially better than random.

---

### Implementation Reference

```python
import torch
import torch.nn.functional as F

# Logits: shape (T, V) = (3, 6)
logits = torch.tensor([
    [ 2.1,  0.3, -1.2,  0.8,  1.5, -0.4],
    [-0.5,  3.2,  0.1, -1.0,  0.2,  1.1],
    [ 1.0,  0.5,  2.8,  0.3, -0.7,  1.4],
], dtype=torch.float32)

# Target token indices: shape (T,)
targets = torch.tensor([0, 1, 2], dtype=torch.long)

# --- Manual computation ---
# Softmax probabilities
probs = F.softmax(logits, dim=-1)
print("Softmax probs (row = position, col = vocab):")
print(probs)
# tensor([[0.4749, 0.0785, 0.0175, 0.1294, 0.2607, 0.0390],
#         [0.0196, 0.7955, 0.0358, 0.0119, 0.0396, 0.0975],
#         [0.1018, 0.0617, 0.6156, 0.0505, 0.0186, 0.1518]])

# Log-probabilities of correct tokens
log_probs = F.log_softmax(logits, dim=-1)
correct_log_probs = log_probs[torch.arange(3), targets]
print("\nLog-prob of correct token per position:", correct_log_probs)
# tensor([-0.7444, -0.2291, -0.4853])

# Per-token NLL
per_token_loss = -correct_log_probs
print("Per-token NLL:", per_token_loss)
# tensor([0.7444, 0.2291, 0.4853])

# Mean loss
loss_manual = per_token_loss.mean()
print(f"\nMean loss (manual): {loss_manual:.4f}")  # 0.4863

# --- PyTorch built-in (should match exactly) ---
loss_builtin = F.cross_entropy(logits, targets)
print(f"Mean loss (F.cross_entropy): {loss_builtin:.4f}")  # 0.4863

# Perplexity
perplexity = torch.exp(loss_builtin)
print(f"Perplexity: {perplexity:.4f}")  # 1.6267
```

**Expected output:**

```
Mean loss (manual):        0.4863
Mean loss (F.cross_entropy): 0.4863
Perplexity:                1.6267
```

---

### Key Takeaways

1. Cross-entropy loss and negative log-likelihood are the same thing for
   single-label classification. For language modelling, each token prediction
   is an independent $V$-class classification problem.

2. The softmax-then-log approach is mathematically correct but numerically
   unstable due to intermediate large/small exponentials. Always use
   `log_softmax` or `cross_entropy` which implement the stable log-sum-exp
   formulation internally.

3. Perplexity is $e^\mathcal{L}$, so a loss improvement of $0.1$ nats
   corresponds to a perplexity reduction that depends on the current value.
   At $\mathcal{L} = 2.0$ (PPL $\approx 7.39$), reducing loss by $0.1$ gives
   PPL $\approx 6.69$ — a drop of $0.70$. Perplexity improvements get cheaper
   (in loss units) as models improve.

4. In teacher-forcing training, the targets at all positions are known in advance
   and the loss is computed over all $T$ positions in a single forward pass —
   there is no sequential dependency during training, only during autoregressive
   inference.
