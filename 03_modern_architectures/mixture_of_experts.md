# Mixture of Experts

## Overview

Mixture of Experts (MoE) decouples model capacity from computational cost: a very large number of parameters can be trained and deployed while only activating a small fraction per token. This is the architecture behind Mixtral, GPT-4 (reportedly), and several Google models (Switch Transformer, Gemini).

---

## Fundamentals

### Q1. What is the core idea behind a Mixture of Experts layer?

**Answer.**

A standard FFN in a transformer maps every token through the same set of parameters. An MoE layer instead maintains $E$ independent expert networks $\{f_1, f_2, \ldots, f_E\}$ and a router (gating network) $G$ that selects which experts to activate for each token.

**Standard MoE forward pass:**

1. **Router.** Compute scores for all experts: $\mathbf{s} = \text{Softmax}(x W_{\text{gate}}) \in \mathbb{R}^E$
2. **Top-k selection.** Select the $k$ experts with highest scores: $\mathcal{T} = \text{top-k}(\mathbf{s})$
3. **Expert computation.** Run the selected experts: $f_e(x)$ for $e \in \mathcal{T}$
4. **Weighted combination.** Output is the weighted sum:
$$y = \sum_{e \in \mathcal{T}} \frac{s_e}{\sum_{j \in \mathcal{T}} s_j} \cdot f_e(x)$$

The normalisation in step 4 ensures the weights for selected experts sum to 1.

**Key property: sparse activation.** If $k \ll E$, each token activates only $k/E$ of all expert parameters. A model with $E = 8$ experts and $k = 2$ activates $25\%$ of FFN parameters per token — enabling $4\times$ the parameters for the same per-token FLOPs.

---

### Q2. What is the difference between "number of experts" and "active experts per token"?

**Answer.**

- **Total experts $E$**: the number of distinct FFN expert networks in the layer. This determines total model parameter count. Larger $E$ increases model capacity.
- **Active experts per token $k$**: the number of experts selected by the router for each token. This determines per-token FLOPs. Smaller $k$ reduces compute.

**The MoE efficiency promise:** Parameter count scales with $E$ (capacity), but FLOPs scale with $k$ (constant regardless of $E$). A dense model with FFN hidden dim $d_{\text{ff}}$ has FLOPs proportional to $d_{\text{ff}}$. An MoE model with $E$ experts of the same size has $E \times d_{\text{ff}}$ parameters but the same FLOPs (for $k = 1$) or $k \times$ FLOPs (for general $k$).

**Mixtral 8x7B example:** 8 experts, $k = 2$, so each token uses 2 experts. Total FFN parameters are $8\times$ that of a dense 7B model, but per-token FLOPs are $2\times$ the dense FFN cost. The model has roughly 45B parameters but computes like a 12B dense model.

---

### Q3. What does the router/gating network do, and what are its failure modes?

**Answer.**

The router is a learned linear projection followed by softmax:

$$\mathbf{s} = \text{Softmax}(x W_{\text{gate}})$$

where $W_{\text{gate}} \in \mathbb{R}^{d \times E}$.

**What it learns:** The router learns to assign different types of tokens (by syntax, semantics, position) to different experts. Studies show that MoE layers develop specialisation: in language models, some experts become specialised for punctuation, numbers, specific language types, or syntactic constructs.

**Failure mode 1: Expert collapse / load imbalance.** The most dangerous failure. If one expert gets slightly higher initial scores, it receives more training examples, improves faster, gets selected more, and the cycle amplifies. Eventually, $k$ out of $E$ experts handle all tokens, and the rest atrophy — wasting parameters. This is the primary challenge in MoE training.

**Failure mode 2: Representation collapse.** If all experts converge to learn similar functions, diversity is lost and the MoE provides no benefit over a single expert.

**Failure mode 3: Token dropping.** With a capacity factor (see Q6), overloaded experts drop tokens. Dropped tokens receive a pass-through (identity or zero), degrading output quality for those tokens.

---

## Intermediate

### Q4. Explain top-k routing. Why is $k = 2$ the most common choice?

**Answer.**

**Top-k routing** selects the $k$ experts with the highest router scores and computes a weighted combination of their outputs.

**Forward pass (top-2):**

1. $\mathbf{s} = \text{Softmax}(x W_{\text{gate}}) \in \mathbb{R}^E$
2. Select indices: $\mathcal{T} = \{e_1, e_2\} = \text{argtop-2}(\mathbf{s})$
3. Renormalise: $\tilde{s}_{e_1} = s_{e_1} / (s_{e_1} + s_{e_2})$, $\tilde{s}_{e_2} = s_{e_2} / (s_{e_1} + s_{e_2})$
4. Output: $y = \tilde{s}_{e_1} f_{e_1}(x) + \tilde{s}_{e_2} f_{e_2}(x)$

**Why $k = 2$ rather than $k = 1$:**

- $k = 1$: the model must make a hard binary choice per token. If the router makes a wrong assignment, there is no fallback. The discrete argmax makes training noisier.
- $k = 2$: smooth interpolation between two experts. The router can express uncertainty by assigning $0.5/0.5$ weights. Training is more stable. Empirically outperforms $k = 1$ at the same FLOPs budget.
- $k > 2$: diminishing returns. Increasing $k$ increases FLOPs without proportional quality gains once $k \geq 3$.

**Why $k = 1$ is used in some settings:** Switch Transformer (Fedus et al., 2021) shows that $k = 1$ works well at very large scale with careful load balancing. At trillion-parameter scales, the simplicity and compute efficiency of $k = 1$ matters.

---

### Q5. What is the auxiliary load balancing loss, and why is it critical?

**Answer.**

Without intervention, top-k routing collapses: a few experts dominate. The load balancing loss explicitly penalises uneven routing to encourage uniform expert utilisation.

**Standard auxiliary loss (Switch Transformer):**

Let $f_e$ be the fraction of tokens routed to expert $e$ in a batch, and $p_e$ be the average router probability for expert $e$:

$$f_e = \frac{\text{number of tokens dispatched to expert } e}{\text{total tokens in batch}}$$

$$p_e = \frac{1}{T} \sum_{t=1}^T s_{t,e}$$

The auxiliary loss:
$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \sum_{e=1}^E f_e \cdot p_e$$

where $\alpha$ is a small coefficient (typically $10^{-2}$ to $10^{-1}$) and $E$ is the number of experts.

**Why this loss works:**
- $f_e$ measures actual token assignment — not differentiable through the argmax
- $p_e$ measures the soft router probability — differentiable
- The product $f_e \cdot p_e$ is minimised when all $f_e$ are equal (uniform assignment). The gradient flows through $p_e$ to train the router toward balance, even though $f_e$ is not differentiable.
- The factor $E$ normalises so that $\mathcal{L}_{\text{aux}} = \alpha$ when perfectly balanced (uniform $f_e = 1/E$, $p_e = 1/E$)

**Tuning $\alpha$:** Too small — collapse occurs. Too large — forced balance degrades quality because the router cannot learn meaningful expert specialisation. Typical values are $\alpha \in [0.01, 0.1]$.

---

### Q6. What is the capacity factor and how does it affect expert load?

**Answer.**

In distributed training, each expert runs on a specific device and can only process a fixed number of tokens per batch. The **capacity factor** $C$ sets this limit:

$$\text{capacity} = C \times \frac{\text{tokens per batch}}{E}$$

If a token is assigned to a full expert (capacity exceeded), it is **dropped**: it either bypasses the MoE layer (residual pass-through) or is routed to the next-best expert.

**Example.** Batch of $T = 1024$ tokens, $E = 8$ experts, $C = 1.25$:
$$\text{capacity} = 1.25 \times 1024/8 = 160 \text{ tokens per expert}$$

Under perfect balance, each expert receives $T/E = 128$ tokens. $C = 1.25$ provides $25\%$ headroom.

**Trade-offs:**

| $C$ value | Effect |
|---|---|
| $C = 1.0$ | Tight budget — any imbalance causes dropping. Efficient but fragile. |
| $C = 1.25$ | Standard — handles moderate imbalance without dropping. |
| $C = 2.0$ | Conservative — almost no dropping but 2x memory/compute overhead for routing buffers. |
| $C \to \infty$ | No dropping, but requires dynamic batching per expert (complex implementation). |

**Token dropping consequences.** Dropped tokens receive the residual (their input $x$ unchanged). For most tokens this is tolerable; for critical tokens (e.g., the subject of a sentence), dropping can significantly degrade generation quality.

---

### Q7. Describe the Mixtral 8x7B architecture. How does it differ from a standard dense MoE?

**Answer.**

Mixtral 8x7B (Jiang et al., 2024) is a sparse MoE model based on the Mistral 7B architecture:

| Component | Mixtral 8x7B |
|---|---|
| Experts per layer | 8 |
| Active experts per token | 2 (top-2) |
| Total layers | 32 |
| Hidden dim | 4096 |
| Expert FFN hidden dim | 14336 |
| Attention | GQA (8 KV heads, 32 Q heads) |
| Position encoding | RoPE |
| Normalisation | RMSNorm |
| Total parameters | $\sim 46.7$B |
| Active parameters per token | $\sim 12.9$B |

**Key design choices:**

1. **Only FFN layers are replaced by MoE.** Attention layers remain dense. This is standard practice — MoE has been less successful for attention, and attention parameters are smaller than FFN parameters.

2. **No auxiliary loss during pretraining.** Mixtral reports using no load balancing loss, relying on natural training dynamics. This differs from Switch Transformer and other models.

3. **Expert choice possible at inference.** The router independently selects 2 experts per token; no capacity constraints are enforced at inference (only at training for compute reasons).

4. **Combines with GQA.** The combination of MoE (more parameters) and GQA (smaller KV cache) allows Mixtral to have large model capacity while maintaining reasonable inference memory.

---

## Advanced

### Q8. What is Expert Choice routing and how does it address load balancing?

**Answer.**

Standard top-k routing is **token choice**: each token independently selects its top-k experts. This can cause load imbalance because multiple tokens may simultaneously prefer the same expert.

**Expert Choice routing** (Zhou et al., 2022) inverts the paradigm: each expert independently selects its top-$k'$ tokens from the batch.

**Forward pass:**
1. Compute all router logits: $S_{te} = x_t W_{\text{gate}} \in \mathbb{R}^{T \times E}$ (token $t$, expert $e$)
2. Each expert $e$ selects the $k' = C \cdot T / E$ tokens with the highest score in column $e$
3. Each selected token is processed by the expert; the output is the score-weighted sum over all experts that selected it

**Load balancing by construction.** Each expert always processes exactly $k'$ tokens. There is no imbalance, no auxiliary loss needed, and no token dropping.

**Drawbacks:**
- **Variable tokens per expert per token.** Some tokens are selected by 0, 1, 2, or more experts — the number of active experts per token is not fixed. This makes it harder to predict FLOPs.
- **Tokens can be dropped.** A token selected by 0 experts is dropped (receives residual pass-through).
- **Training instability.** Because different tokens receive different numbers of experts, the effective learning rate varies per token.

**When to use which:**
- Token choice (top-k): standard, predictable FLOPs per token, requires load balancing loss
- Expert choice: automatic load balance, but variable per-token compute and more complex implementation

---

### Q9. How is expert parallelism implemented, and what communication is required?

**Answer.**

In expert parallelism (EP), experts are distributed across $P$ devices: device $p$ holds experts $\{(p-1)E/P + 1, \ldots, pE/P\}$.

**Communication pattern:**

Step 1: **All-to-all (dispatch).** Each device has a batch of tokens. After the router assigns tokens to experts, tokens destined for experts on other devices must be sent there. This is an all-to-all communication where each device sends a different subset of its tokens to each other device.

Step 2: **Expert computation.** Each device runs its $E/P$ expert networks on the received tokens. Fully local.

Step 3: **All-to-all (combine).** Expert outputs must be returned to the originating device for the weighted sum. A second all-to-all sends outputs back.

**Communication volume.** Each token generates $k$ messages (one per selected expert). With $T$ tokens per device and $k = 2$ active experts:
- Expected tokens sent off-device per step: $T \times k \times (1 - 1/P) = T \times 2 \times (P-1)/P$
- Each token's representation is $d$ floats $\times$ 2 bytes = $2d$ bytes

For $d = 4096$, $T = 1024$, $P = 8$: $\approx 1024 \times 2 \times 7/8 \times 8192 \approx 14$ MB per device per all-to-all. At NVLink bandwidth ($600$ GB/s bidirectional), this takes $\sim 24\ \mu$s — a small fraction of the compute time for large models, but grows with $T$ and $d$.

**EP combined with tensor parallelism.** Production systems often use EP across nodes (slower interconnects) and TP within nodes (NVLink). The all-to-all for EP uses slow inter-node bandwidth, which is the key scalability bottleneck for very large MoE models.
