# Learning Rate Schedules

## Overview

The learning rate schedule is one of the most consequential hyperparameters in training large language models. Poor scheduling causes training instability, wasted compute, or suboptimal convergence. The original Transformer introduced a specific schedule with a warmup phase; modern practice has evolved to use cosine annealing with warmup. Understanding the mechanics and motivation for each approach, along with the relationship between learning rate and batch size, is expected in ML engineering interviews.

---

## Tier 1: Fundamentals

### Q1. What is the original Transformer learning rate schedule? Write the formula and explain each component.

**Answer.**

Vaswani et al. (2017) used the following schedule:

$$\text{lr}(\text{step}) = d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$

**Component analysis:**

**1. $d_{\text{model}}^{-0.5}$ scaling:**
Scales the learning rate inversely with the square root of the model dimension. Larger models have more parameters and smaller individual parameter gradients (due to residual scaling), so they need a proportionally adjusted learning rate. This normalisation makes the schedule transferable across model sizes.

**2. $\min(\ldots)$ — two-phase schedule:**

The schedule has two regimes based on which argument is smaller:

**Phase 1 (warmup): $\text{step} < \text{warmup\_steps}$**

When $\text{step}$ is small:
- $\text{step}^{-0.5}$ is large
- $\text{step} \cdot \text{warmup\_steps}^{-1.5}$ is small

The minimum selects the linear warmup term: lr increases linearly with step count.

$$\text{lr} \propto \text{step}$$

**Phase 2 (decay): $\text{step} > \text{warmup\_steps}$**

Once step exceeds warmup:
- $\text{step} \cdot \text{warmup\_steps}^{-1.5}$ exceeds $\text{step}^{-0.5}$
- The minimum selects the inverse square root decay

$$\text{lr} \propto \text{step}^{-0.5}$$

**Numerical example** ($d_{\text{model}} = 512$, $\text{warmup\_steps} = 4000$):

- Peak LR: at step = 4000: $512^{-0.5} \times 4000^{-0.5} = 0.044 \times 0.0158 \approx 7 \times 10^{-4}$
- At step 8000: $512^{-0.5} \times 8000^{-0.5} \approx 5 \times 10^{-4}$ (decaying)
- At step 40000: $512^{-0.5} \times 40000^{-0.5} \approx 2.2 \times 10^{-4}$

---

### Q2. What is learning rate warmup and why is it necessary for training Transformers?

**Answer.**

**Definition:** A warmup period is an initial phase of training where the learning rate is increased from a small value (often 0 or $10^{-7}$) to a target learning rate over a number of steps.

**Why warmup is necessary:**

**1. Unstable gradient estimates early in training.**

At the start of training, model weights are random and the loss landscape is chaotic. Gradients are noisy and unreliable. A large learning rate early on can cause the model to overshoot good regions of the loss landscape and diverge (loss goes to NaN or very large values).

**2. Adaptive optimiser warm-up (Adam-specific).**

Adam and AdaGrad maintain running estimates of the gradient mean ($m_t$) and variance ($v_t$). Early in training, these estimates are based on very few observations and are unreliable:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

For small $t$, the bias correction terms $1 - \beta_1^t$ and $1 - \beta_2^t$ are large, and the effective update magnitude is unstable. Starting with a small learning rate prevents large, poorly-informed updates during this window.

**3. Large models amplify instability.**

For very deep models with many parameters, a single bad update can shift the model into a region where all gradients are poor, requiring expensive recovery or restarting from an earlier checkpoint. Warmup reduces this risk substantially.

**Empirical evidence:** Liu et al. (2020) showed that variance of gradients is largest at the beginning of training when Adam's $v_t$ estimate is unreliable. Warmup reduces the variance of parameter updates during this period.

**Common warmup configurations:**
- Original Transformer: 4,000 steps of linear warmup
- BERT: 10,000 steps (up to 1% of total training steps)
- GPT-3: 375M tokens (~ a few thousand steps)
- Modern practice: 1–5% of total training steps for warmup

---

### Q3. Describe cosine annealing with warmup. Why has it become the dominant schedule for LLM training?

**Answer.**

**Cosine annealing with warmup:**

The schedule proceeds in two phases:

**Phase 1 — Linear warmup** (steps 0 to $T_{\text{warm}}$):

$$\text{lr}(t) = \text{lr}_{\text{max}} \cdot \frac{t}{T_{\text{warm}}}$$

**Phase 2 — Cosine decay** (steps $T_{\text{warm}}$ to $T_{\text{total}}$):

$$\text{lr}(t) = \text{lr}_{\text{min}} + \frac{1}{2}\!\left(\text{lr}_{\text{max}} - \text{lr}_{\text{min}}\right)\!\left(1 + \cos\!\left(\pi \cdot \frac{t - T_{\text{warm}}}{T_{\text{total}} - T_{\text{warm}}}\right)\right)$$

where $\text{lr}_{\text{min}}$ is a small final learning rate (often 10% of $\text{lr}_{\text{max}}$, or even 0).

**Why cosine annealing dominates:**

1. **Smooth decay:** Unlike step-wise decay (sudden drops), cosine annealing decays smoothly. The gradient of the schedule is zero at the beginning and end of the decay phase, providing a gentle transition.

2. **Predictable budget:** The schedule is parameterised by total training steps, which aligns naturally with compute budgets. You set the budget first, then define the schedule.

3. **Strong empirical performance:** Cosine annealing consistently outperforms step-decay and the original Transformer schedule in large-scale experiments (e.g., Chinchilla paper).

4. **End-of-training stabilisation:** The decaying LR in the final phase allows the model to settle into a local minimum rather than bouncing around. The last few percent of training often provide disproportionate performance gains.

**Typical hyperparameters:**
- $\text{lr}_{\text{max}} \approx 3 \times 10^{-4}$ for medium-scale LLMs (adjusted with batch size)
- $\text{lr}_{\text{min}} = 0.1 \times \text{lr}_{\text{max}}$ or $= 0$
- $T_{\text{warm}} = 1\%$–$5\%$ of $T_{\text{total}}$

---

### Q4. What is linear learning rate decay? When is it preferred over cosine annealing?

**Answer.**

**Linear decay:**

After warmup, the learning rate decreases linearly to zero:

$$\text{lr}(t) = \text{lr}_{\text{max}} \cdot \max\!\left(0,\; 1 - \frac{t - T_{\text{warm}}}{T_{\text{total}} - T_{\text{warm}}}\right)$$

**When linear decay is used:**

- **BERT and early encoder models** used linear decay to the final fine-tuning LR because the short training windows (3–5 epochs) made cosine annealing and linear decay roughly equivalent.
- **Some research ablations** prefer linear decay for its interpretability — the schedule is a simple function with no cosine oscillation.
- **Warmup-only schedules** (Chinchilla paper, many modern LLMs): Train with warmup + constant LR for most of training, then apply a brief linear or cosine decay in the final phase. This is sometimes called "warmup-stable-decay" (WSD).

**Warmup-Stable-Decay (WSD) schedule:**

Recently popularised by MiniCPM (Hu et al., 2024) and several other works:

1. **Warmup phase** (1–5% of training): linear warmup to $\text{lr}_{\text{max}}$
2. **Stable phase** (90–95% of training): constant $\text{lr} = \text{lr}_{\text{max}}$
3. **Decay phase** (final 5–10%): linear or cosine decay to $\text{lr}_{\text{min}}$

Advantage: the stable phase allows easy checkpointing and continuation — you can always extend training by continuing the stable phase, whereas cosine annealing requires knowing the total training steps in advance.

---

## Tier 2: Intermediate

### Q5. What is the relationship between learning rate and batch size? State the linear scaling rule and when it fails.

**Answer.**

**The linear scaling rule (Goyal et al., 2017):**

When you multiply the batch size by $k$, multiply the learning rate by $k$:

$$\text{lr}_{\text{new}} = k \cdot \text{lr}_{\text{base}}, \quad B_{\text{new}} = k \cdot B_{\text{base}}$$

**Intuition:**

With SGD, one gradient step on a batch of size $B$ estimates the gradient with noise $\propto 1/\sqrt{B}$. If you use a batch of size $kB$, the gradient estimate is $\sqrt{k}$ times more accurate, so you can take a $\sqrt{k}$ times larger step (square root rule). Alternatively, with a batch of $kB$, each step "corresponds to" $k$ smaller steps — so using the same number of gradient updates is like doing $k$ times fewer small steps, suggesting the learning rate should be $k$ times larger to cover the same parameter space per epoch (linear rule).

Empirically, the **linear scaling rule** works well for moderate batch size increases ($k \leq 32$).

**Why the linear rule fails for very large batches:**

1. **The gradient noise floor:** For sufficiently large batches, the gradient estimate is essentially noise-free. The "noise benefit" of large batches saturates, and further doubling the batch size doesn't require proportionally larger learning rates.

2. **The critical batch size** $B_{\text{crit}}$: McCandlish et al. (2018) showed there is a critical batch size beyond which more gradient samples don't significantly reduce the variance of the gradient. For $B \gg B_{\text{crit}}$, the square root rule applies ($\text{lr} \propto \sqrt{B}$), not the linear rule.

3. **Warmup at large batch sizes:** Large learning rates with large batches are numerically unstable early in training. The warmup period must be extended proportionally with batch size.

**Practical guidelines:**

| Batch size change | Learning rate adjustment |
|---|---|
| $\times 2$ (moderate) | $\times 2$ (linear rule) |
| $\times 8$ (moderate) | $\times 8$ or $\times 2.83$ ($\sqrt{8}$) — experiment |
| $\times 32$ (large) | $\times \sqrt{32} \approx 5.7$ is safer |
| $\times 256$ (very large) | Likely need the square root rule + longer warmup |

---

### Q6. How do you tune the learning rate for a large model when you cannot afford to do a full hyperparameter sweep?

**Answer.**

Full learning rate sweeps are prohibitively expensive for large models. Practitioners use several techniques:

**1. Transfer from smaller models (Maximal Update Parameterisation, muP):**

Yang et al. (2021) showed that with a specific parameterisation ($\mu$P), the optimal learning rate is approximately constant across model sizes. You can:

1. Train a small proxy model (e.g., 100M parameters) at many learning rates
2. Find the optimal LR for the small model
3. Use approximately the same LR for the 10B parameter model

Under standard parameterisation (NTK/default PyTorch), optimal LR decreases with width — you cannot transfer hyperparameters. Under $\mu$P, optimal LR is width-invariant.

**2. Learning rate range test (Smith, 2017):**

Run training for a short number of steps while linearly increasing the learning rate from a very small value to a large value. Plot the loss vs. learning rate:

- Loss decreasing: LR is too small
- Loss at minimum: optimal region
- Loss increasing: LR is too large

Choose a learning rate slightly below the minimum loss point, or at the beginning of the loss decline. Takes ~5% of a full training run.

**3. Rule-of-thumb baselines:**

For Adam optimizer with cosine decay:
- Small models (100M): $\text{lr}_{\text{max}} \approx 3 \times 10^{-4}$
- Medium models (1B): $\text{lr}_{\text{max}} \approx 2 \times 10^{-4}$
- Large models (7B): $\text{lr}_{\text{max}} \approx 1 \times 10^{-4}$
- Very large models (70B+): $\text{lr}_{\text{max}} \approx 5 \times 10^{-5}$

These are common starting points; $\mu$P-based transfer is more principled.

**4. Monitoring and adaptation:**

During training, monitor:
- Loss curve for sudden spikes (LR too high)
- Gradient norm for exploding values
- Parameter update-to-weight ratio (should be ~$10^{-3}$)

If spikes occur: reduce LR by 10x and restart from the last clean checkpoint.

---

## Tier 3: Advanced

### Q7. Derive the Adam update rule and explain why adaptive learning rates interact with the warmup phase.

**Answer.**

**Adam (Kingma & Ba, 2014):**

Given gradient $g_t$ at step $t$:

**1. Update moment estimates:**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment — gradient mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment — gradient variance)}$$

**2. Bias correction:**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

At step $t = 1$: $1 - \beta_1^1 = 1 - 0.9 = 0.1$ (strong correction since $m_1$ underestimates the true first moment).

**3. Parameter update:**

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

where $\alpha$ is the learning rate.

**Why bias correction is critical early in training:**

Without bias correction, $m_t \approx (1-\beta_1) g_t$ for small $t$ (since $m_0 = 0$). The effective update would be:

$$\theta_t \approx \theta_{t-1} - \frac{\alpha}{\sqrt{(1-\beta_2)g_t^2} + \epsilon} \cdot (1-\beta_1) g_t = \theta_{t-1} - \frac{\alpha (1-\beta_1)}{\sqrt{1-\beta_2}\,|g_t| + \epsilon} \cdot g_t$$

For typical values $\beta_1 = 0.9$, $\beta_2 = 0.999$:

$$\text{scale factor} = \frac{1-\beta_1}{\sqrt{1-\beta_2}} = \frac{0.1}{\sqrt{0.001}} \approx \frac{0.1}{0.0316} \approx 3.16$$

The bias correction makes the effective step size 3.16x larger than it would be without correction. This is the "Adam bias correction spike" — at step 1, Adam takes a disproportionately large step.

**Interaction with warmup:**

The warmup period reduces $\alpha$ to near zero at step 0. This counteracts the large bias-corrected step size at early training. As training proceeds:
- Bias correction factors approach 1 (bias shrinks)
- Learning rate reaches its target value

The two effects — bias correction decreasing, learning rate increasing — are designed to produce smooth, stable early training dynamics.

**AdamW modification:**

Adam with weight decay decouples the weight decay term from the gradient update:

$$\theta_t = \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

Standard L2 regularisation applies weight decay inside the adaptive gradient, which has the perverse effect of making weight decay larger for parameters with small gradients. AdamW decouples this, applying a uniform weight decay $\lambda$ to all parameters regardless of their gradient magnitude.

---

### Q8. What are "loss spikes" during LLM training and what causes them? How do practitioners detect and recover from them?

**Answer.**

**Loss spikes:** Sudden, large increases in training loss (typically 2–10x the previous loss value) that occur during the stable phase of training.

**Documented occurrence:** Loss spikes were observed during GPT-3, PaLM, LLaMA, and most large-scale training runs.

**Causes:**

**1. Pathological training batches:**
Occasionally, a mini-batch contains text that is highly anomalous relative to the training distribution — very long sequences, unusual character patterns, degenerate formatting, or near-duplicate noisy data. The model's gradient on this batch is large and in an unusual direction, causing a spike.

**2. Learning rate too high:**
When the learning rate is near its maximum, the model is making large parameter moves. A combination of slightly-too-high LR and a bad batch causes overshooting.

**3. Numerical precision issues:**
In mixed-precision training, FP16 can overflow or underflow for large gradients. This produces NaN or Inf gradient values that contaminate the Adam moment estimates, causing persistent instability.

**4. Data pipeline issues:**
Corrupted data appearing at a specific step (e.g., a file with NaN values, a batch of extremely long sequences that don't fit cleanly into the sequence length).

**Detection:**

Monitor:
- **Loss** — value and first derivative; alert on sudden increases > 2x
- **Gradient norm** — if it exceeds 5–10x the typical value, a spike may be imminent
- **Parameter update norm** — abnormally large updates signal trouble
- **Learning rate schedule** — verify schedule is as expected

**Recovery strategies:**

1. **Rollback and skip:** Save checkpoints every N steps. On detecting a spike, roll back to the most recent clean checkpoint, identify and remove the offending batch from the data loader, and continue training.

2. **Reduce LR temporarily:** After rollback, reduce the learning rate by a factor of 10 for a few hundred steps before resuming the original schedule.

3. **Gradient clipping:** Clip gradients to a maximum norm (typically 1.0). This limits the magnitude of any individual update, preventing single bad batches from causing large parameter moves:

   $$g \leftarrow g \cdot \min\!\left(1, \frac{C}{\|g\|_2}\right)$$

4. **Data filtering:** Increase the aggressiveness of data cleaning to reduce the probability of pathological batches. Heuristics: remove sequences with extreme perplexity under a small reference model, remove sequences with unusual character-to-token ratios.

5. **Loss spike tolerance:** For very large runs, some teams accept occasional spikes and do not roll back if the loss self-recovers within 1000 steps. Self-recovery is common if gradient clipping is in place.

**Prevention:**

- Use gradient clipping (nearly universal)
- Monitor the data pipeline for corruption
- Use BF16 instead of FP16 — BF16 has a much larger dynamic range (same as FP32), preventing overflow/underflow
- Implement automatic checkpoint-and-rollback systems
