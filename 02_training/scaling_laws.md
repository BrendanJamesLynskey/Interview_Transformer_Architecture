# Scaling Laws

## Overview

Scaling laws describe how model performance varies with model size, dataset size, and compute budget. The Kaplan et al. (2020) and Hoffmann et al. (Chinchilla, 2022) papers are among the most influential results in modern AI, directly shaping multi-billion dollar training decisions. Understanding the power law relationships, the optimal compute allocation, and the gap between Kaplan and Chinchilla is critical for any research or engineering role working on LLMs.

---

## Tier 1: Fundamentals

### Q1. What is a scaling law? State the Kaplan et al. power law relationship.

**Answer.**

A **scaling law** is an empirical relationship showing how a model's performance (typically measured by loss) changes as key resources are scaled.

**Kaplan et al. (2020) — "Scaling Laws for Neural Language Models":**

The key finding is that language model loss follows **power laws** in:

**Model size (parameters $N$):**

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

**Dataset size (tokens $D$):**

$$L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

**Compute (FLOPs $C$, with optimal model size):**

$$L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.048$$

where $N_c$, $D_c$, $C_c$ are data-dependent constants.

**In plain language:**

- Doubling the number of parameters reduces loss by a factor of $2^{0.076} \approx 1.054$ (5.4% reduction)
- Doubling the training tokens reduces loss by $2^{0.095} \approx 1.069$ (6.9% reduction)
- Doubling compute reduces loss by $2^{0.048} \approx 1.034$ (3.4% reduction at fixed data-to-parameter ratio)

**Key observations from Kaplan et al.:**

1. Performance improves smoothly and predictably with scale — no phase transitions or plateaus
2. Performance is more sensitive to model size than to dataset size (for the compute-optimal regime)
3. Optimal allocation (given fixed compute $C$): spend most of the budget on model parameters, train on a relatively small number of tokens

---

### Q2. What did Chinchilla (Hoffmann et al., 2022) revise? State the compute-optimal relationship.

**Answer.**

**The Chinchilla paper's finding:** Kaplan et al.'s models were under-trained. GPT-3 (175B parameters, ~300B tokens) was trained with far too many parameters and far too few tokens given its compute budget.

**Chinchilla's compute-optimal relationship:**

For a given compute budget $C$ (measured in FLOPs), the optimal number of parameters $N^*$ and training tokens $D^*$ satisfy:

$$N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

More precisely:

$$N^* \approx \frac{C}{6D^*}, \quad D^* \approx 20 \cdot N^*$$

The key formula is the **1:20 ratio**:

$$\text{Optimal: } D^* = 20 \times N^*$$

Training a model of $N$ parameters requires approximately $20N$ tokens for compute-optimal training.

**Why this differs from Kaplan:**

Kaplan et al. found $\alpha_N > \alpha_D$ (loss is more sensitive to model size than data size), which suggested: given fixed compute, use a larger model trained on fewer tokens. This led to GPT-3's design.

Chinchilla found that when the learning rate schedule and training budget are both properly optimised, $\alpha_N \approx \alpha_D$ — both scaling equally. Therefore, the compute-optimal allocation is equal scaling of model size and data.

**Chinchilla validation:** The 70B Chinchilla model (trained on 1.4T tokens) matched or exceeded GPT-3 (175B, 300B tokens) on most benchmarks — using the same compute but 2.5x fewer parameters and 4.5x more data.

---

### Q3. What are the practical implications of Chinchilla scaling laws for industry?

**Answer.**

**1. Reframing what "optimal" means:**

Pre-Chinchilla: large model + relatively small data = "big model"
Post-Chinchilla: optimal model + 20× as many tokens = "compute-optimal model"

LLaMA (Touvron et al., 2023) explicitly adopted the Chinchilla perspective: train a 7B model on 1T tokens (100× Chinchilla-optimal) so that the deployed model is smaller (faster inference) while trained to maximum capability.

**2. Inference cost matters:**

Training cost is paid once. Inference cost is paid for every API call. A Chinchilla-optimal model achieves a given capability level with a smaller model than a Kaplan-optimal model — significantly reducing inference cost at the same capability level.

For a business deploying an API, using a 7B model instead of a 175B model for the same benchmark score represents ~25x lower inference compute.

**3. Data is the bottleneck:**

Chinchilla implies that high-quality training data is as valuable as compute. This has driven investment in data curation, synthetic data generation, and data filtering pipelines.

**4. "Training beyond Chinchilla optimal":**

The Chinchilla optimum is for achieving the best loss per FLOPs of training. But if inference is cheaper than training, it can make sense to train a smaller model on more tokens than Chinchilla-optimal — you spend more training compute to produce a model that requires less inference compute.

LLaMA's design philosophy: 7B model on 1.4T tokens uses more training FLOPs than Chinchilla-optimal for 7B, but produces a model that is cheap to run at inference.

**5. Data quality over quantity:**

Post-Chinchilla, research has shown that data quality matters enormously. 1 trillion tokens of carefully curated text can outperform 5 trillion tokens of raw web data. This motivates datasets like FineWeb, DCLM, and RedPajama-V2 with extensive quality filtering.

---

## Tier 2: Intermediate

### Q4. Derive the compute-optimal allocation. Starting from the loss surface over $(N, D)$, derive why $D^* \propto N^*$ is optimal.

**Answer.**

**Empirical loss function:**

Chinchilla models the loss as a function of model parameters $N$ and training tokens $D$:

$$L(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + L_{\text{irr}}$$

where $L_{\text{irr}}$ is the irreducible loss (entropy of the data), and empirically:

$$A = 406.4, \quad \alpha = 0.3392, \quad B = 410.7, \quad \beta = 0.2849$$

(These are the fitted values from Chinchilla.)

**Compute budget constraint:**

The FLOPs required to train a model of size $N$ on $D$ tokens scales approximately as:

$$C \approx 6ND$$

(The factor 6 accounts for: 2 FLOPs per parameter per forward pass, $\times 3$ for forward + backward + activation recomputation; approximately valid for large Transformers.)

**Optimisation problem:**

Minimise $L(N, D) = A/N^\alpha + B/D^\beta$ subject to $6ND = C$ (constant compute).

Using the constraint: $D = C / (6N)$. Substitute into $L$:

$$L(N) = \frac{A}{N^\alpha} + \frac{B \cdot (6N)^\beta}{C^\beta}$$

**Differentiate and set to zero:**

$$\frac{dL}{dN} = -\frac{A\alpha}{N^{\alpha+1}} + \frac{6^\beta B \beta N^{\beta-1}}{C^\beta} = 0$$

$$\frac{A\alpha}{N^{\alpha+1}} = \frac{6^\beta B \beta N^{\beta-1}}{C^\beta}$$

$$A\alpha C^\beta = 6^\beta B \beta N^{\alpha + \beta}$$

$$N^* = \left(\frac{A\alpha C^\beta}{6^\beta B \beta}\right)^{1/(\alpha+\beta)} = \left(\frac{A\alpha}{6^\beta B \beta}\right)^{1/(\alpha+\beta)} C^{\beta/(\alpha+\beta)}$$

**Scaling exponent:**

$$N^* \propto C^{\beta / (\alpha + \beta)}$$

With $\alpha = 0.3392$, $\beta = 0.2849$:

$$\frac{\beta}{\alpha + \beta} = \frac{0.2849}{0.6241} \approx 0.4565$$

Similarly for $D^*$: by symmetry:

$$D^* \propto C^{\alpha / (\alpha + \beta)} = C^{0.3392/0.6241} \approx C^{0.5435}$$

**The key ratio:**

$$\frac{D^*}{N^*} = \frac{C^{\alpha/(\alpha+\beta)}}{C^{\beta/(\alpha+\beta)}} \cdot \text{constant} = C^{(\alpha-\beta)/(\alpha+\beta)} \cdot \text{constant}$$

Since $\alpha \approx \beta$ (0.3392 ≈ 0.2849), this ratio is approximately constant — both $N^*$ and $D^*$ scale roughly as $C^{0.5}$. Numerically, the constant ratio works out to approximately $D^* \approx 20 N^*$.

**Why the "20x rule" is approximate:**

The exact ratio depends on the empirical constants $A, B, \alpha, \beta$. Different fitting procedures on different model families give slightly different constants. The "20x" figure should be understood as a rough guide: some analyses suggest 15x–30x depending on the data distribution and model architecture.

---

### Q5. How do you extrapolate scaling laws to predict the performance of a model before training it?

**Answer.**

**The extrapolation approach:**

Train several smaller models (e.g., 10M, 100M, 500M parameters) on the same data for the full planned token budget. Fit the power law:

$$L(N) = \frac{A}{N^\alpha} + L_{\text{irr}}$$

Extrapolate to the target model size (e.g., 10B parameters).

**Step-by-step procedure:**

1. **Choose a set of proxy model sizes** spanning 1–2 orders of magnitude below the target
2. **Train each model to the same token budget** (or to Chinchilla-optimal) on a representative subset of the training data
3. **Record final validation loss** for each model size
4. **Fit a power law** on a log-log scale (linear regression in log space):

$$\log L = -\alpha \log N + \log A + \text{correction for } L_{\text{irr}}$$

5. **Extrapolate** to the target model size: plug $N_{\text{target}}$ into the fitted equation

**Confidence intervals:**

The power law fit typically has tight error bars in log space. An extrapolation of 10× model size (one order of magnitude) is generally considered reliable; extrapolation of 100× or more carries larger uncertainty.

**Important caveats:**

1. **Architecture changes break extrapolation.** A switch from MHA to MoE, or a change in data mixture, can cause the predicted loss to be off by 10–20%.

2. **Data distribution matters.** The scaling law is fit on the same data distribution used for the large run. If the final model uses different data filtering, the prediction may be off.

3. **Emergent capabilities are not captured.** Scaling laws predict smooth loss, but task performance (accuracy on benchmarks) can exhibit sharp transitions as model size increases ("emergent abilities"). Loss-based extrapolation doesn't predict these.

4. **Irreducible loss $L_{\text{irr}}$ must be estimated.** If $L_{\text{irr}}$ is not correctly accounted for, the power law fit will be biased. In practice, $L_{\text{irr}}$ is estimated from the asymptote of loss vs. training on very large data.

---

## Tier 3: Advanced

### Q6. Compare Kaplan (2020) and Chinchilla (2022) scaling law derivations. What methodological differences explain the different conclusions?

**Answer.**

**Three key methodological differences:**

**1. Learning rate schedule:**

Kaplan et al. used a **fixed schedule**: train each model to convergence with a schedule appropriate for that model size. When model size increases, they kept the schedule largely fixed. Crucially, they did not re-tune the schedule for each model-size/dataset-size combination.

Hoffmann et al. tuned the learning rate schedule **for each $(N, D)$ combination** separately. The Kaplan models were using schedules that were suboptimal for the amount of data they were trained on — under-training the larger models.

**2. IsoFLOP analysis:**

Kaplan held either $N$ or $D$ fixed and varied the other. They did not perform systematic isoFLOP experiments (hold $C$ fixed, vary the $(N, D)$ allocation).

Chinchilla explicitly did **IsoFLOP analysis**: for each compute budget $C$, train models with many different $(N, D)$ combinations satisfying $6ND = C$, and find the minimum loss. This is the right experiment to determine compute-optimal allocation.

**3. Range of model sizes:**

Kaplan's experiments used models from ~1M to ~1.5B parameters. Chinchilla extended this to 70B parameters with better data. The Kaplan regime was largely data-limited (models were undertrained), while Chinchilla's larger, better-trained models revealed different scaling behaviour.

**Why the different conclusions:**

In Kaplan's regime (models undertrained on small datasets):
- Adding more parameters is very efficient (each parameter sees many tokens, is well-trained)
- Adding more data shows diminishing returns (the model is already learning all it can from the data)
- Conclusion: scale parameters more aggressively

In Chinchilla's regime (models trained to compute-optimal):
- Both parameters and data contribute roughly equally to loss reduction
- Conclusion: scale both equally

**The unifying view:**

Both papers are correct within their respective experimental conditions. The discrepancy arises because Kaplan was measuring efficiency in a data-limited regime, while Chinchilla measured it in the compute-limited regime. For practitioners planning actual large model training, Chinchilla's isoFLOP methodology is the appropriate framework.

---

### Q7. Beyond Chinchilla: what extensions and challenges to the simple power law picture have emerged?

**Answer.**

**1. The Chinchilla equations don't account for inference cost.**

Chinchilla optimises for minimum training loss given compute. But in production, a 7B model trained on 2T tokens may serve the same capability as a 70B model trained on 200B tokens — at 10× lower inference cost. "Inference-optimal" scaling (Sardana & Frankle, 2023) shows that if you plan to run the model for $T$ tokens of inference, the optimal training allocation changes significantly:

$$\text{Optimal } N = \arg\min_N \left[ C_{\text{train}}(N) + T \cdot C_{\text{infer}}(N) \right]$$

where $C_{\text{infer}}(N) \propto N$ (inference cost scales with model size). The result: smaller models trained on more data are preferred when the inference budget is large.

**2. Data quality creates a separate dimension.**

Simple scaling laws assume i.i.d. sampling from a fixed data distribution. In practice, data quality (filtering duplicates, removing noisy text, balancing domains) interacts with the scaling law constants. Dolma (2024) and FineWeb (2024) papers show that at the same model size and token count, a 10% improvement in data quality can match a 2× increase in model size.

**3. Emergent abilities don't follow smooth power laws.**

Wei et al. (2022) documented "emergent abilities" — capabilities that appear suddenly at threshold model scales rather than improving smoothly. Chain-of-thought reasoning, arithmetic in context, and several other benchmarks show sharp transitions. This is in tension with the smooth power law prediction.

Counter-argument (Schaeffer et al., 2023): Many "emergent" transitions are artefacts of the metrics used. If you measure the same capabilities with continuous metrics (calibration, log-probability), the transitions become smooth. The discontinuities arise from threshold metrics (exact match accuracy), not the underlying model capabilities.

**4. Beyond single-task loss:**

All scaling laws measure next-token prediction loss on a fixed distribution. They predict perplexity well. They predict benchmark performance poorly for specific tasks. New work (BIG-bench, HELM) attempts to characterise multi-task scaling, but no unified framework yet exists.

**5. Architecture-specific scaling:**

Mixture-of-Experts (MoE) models (Mixtral, Gemma-MoE) have different scaling laws than dense Transformers because a MoE model with $N_{\text{total}}$ parameters activates only $N_{\text{active}} \ll N_{\text{total}}$ per token. The relevant quantity for FLOPs is $N_{\text{active}}$, but for storage and memory it is $N_{\text{total}}$. Artetxe et al. (2021) showed MoE models follow similar power laws in $N_{\text{active}}$ but can achieve better perplexity per parameter than dense models at the same $N_{\text{total}}$.
