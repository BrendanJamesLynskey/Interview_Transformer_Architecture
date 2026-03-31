# Constitutional AI: Interview Questions

Constitutional AI (CAI) is Anthropic's method for training AI systems to be helpful, harmless, and honest using a set of explicit principles rather than relying solely on human feedback at every step. It introduces AI-generated feedback (RLAIF) as a scalable alternative to human annotation.

---

## Fundamentals

### Q1. What is Constitutional AI, and what problem does it solve?

**Answer.**

Constitutional AI (Bai et al., 2022, Anthropic) addresses two limitations of standard RLHF:

**1. Scalability of human feedback.** Collecting human preference data is expensive and slow. Training models to avoid harmful behaviours requires annotators to read and evaluate harmful content — costly both financially and in terms of annotator wellbeing. CAI replaces much of this human feedback with AI-generated feedback, guided by a written set of principles (the "constitution").

**2. Opacity of human feedback.** In standard RLHF, the model's values are implicit in the annotation guidelines given to human raters. It is difficult to inspect or audit what values the reward model has learned. A written constitution makes the model's values explicit, interpretable, and adjustable by editing a text file rather than by retraining a reward model from scratch.

CAI works in two stages:

- **SL-CAI (Supervised Learning-CAI):** The model critiques and revises its own potentially harmful responses using principles from the constitution, producing a fine-tuning dataset of self-corrected responses.
- **RL-CAI (Reinforcement Learning-CAI):** A preference model is trained on AI-generated preference labels (RLAIF). The language model is then fine-tuned using RL against this AI preference model.

---

### Q2. What is a constitution in the context of CAI?

**Answer.**

A constitution is a set of written principles specifying what values the AI system should uphold. It functions as a rubric the AI uses to evaluate and revise its own outputs.

Example principles from Anthropic's published constitution:

- "Choose the response that is least likely to contain harmful or unethical content."
- "Choose the response that is more likely to be true and to not exaggerate claims."
- "Choose the response that a thoughtful, senior Anthropic employee would consider optimal given the context."
- "Choose the response that is most supportive and encouraging of human autonomy and self-determination."

The constitution is intentionally diverse, covering different ethical frameworks (consequentialist, deontological, virtue-based), different harm categories, and different social concerns. This diversity is deliberate: no single ethical principle is sufficient, and adversarial robustness requires covering many different ways a response could go wrong.

A critical design property: the constitution can be updated without retraining the model from scratch. You can change which principles are active by editing the prompts given to the critic during the CAI pipeline. This makes CAI more transparent and adjustable than a reward model that has absorbed principles implicitly from annotation guidelines.

---

### Q3. Describe the critique-revision loop in SL-CAI step by step.

**Answer.**

The SL-CAI stage uses a multi-step prompted process to get the model to self-correct potentially harmful responses.

**Step 1: Initial response generation.**
The model is given a potentially harmful or adversarial prompt (red-teaming prompts, e.g., asking for instructions for dangerous activities) and generates an initial response without additional safety conditioning. This response may contain harmful content.

**Step 2: Critique.**
The model is prompted to critique its own response using a principle sampled from the constitution. For example:

```
[Original conversation and assistant response]

Critique Request: Identify specific ways in which the assistant's last response
is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.

Critique: ...
```

The model generates a self-critique identifying problems with its initial response.

**Step 3: Revision.**
The model is prompted to revise the response based on its critique:

```
[Original conversation, assistant response, critique]

Revision Request: Please rewrite the assistant response to remove all harmful,
unethical, racist, sexist, toxic, dangerous, or illegal content.

Revised Response: ...
```

**Step 4: Repeat (optional).**
The critique-revision cycle can be applied multiple times, each time potentially using a different constitutional principle. In practice, 1–2 revisions are sufficient for most prompts.

**Step 5: Fine-tuning.**
The (prompt, final revised response) pairs form a supervised fine-tuning dataset. The model is fine-tuned on these pairs, learning to produce more aligned responses directly without needing the critique-revision scaffold at inference time.

This process is analogous to distillation: the multi-step critique-revision process is a slow, deliberate reasoning procedure; SFT on its outputs trains the model to produce the end result directly.

---

### Q4. What is RLAIF (Reinforcement Learning from AI Feedback)?

**Answer.**

RLAIF is the preference labelling stage of RL-CAI. Instead of having human annotators compare pairs of responses, a language model (acting as an AI labeller) is prompted to express a preference between two responses using the constitution:

```
Consider the following conversation and two possible responses.

[Conversation and two candidate responses A and B]

Which response is less harmful according to the following principle:
"Choose the response that is least likely to encourage or assist in illegal activity."

Response: (A) or (B)
```

The AI labeller assigns a preference label to each pair. These AI-generated labels are used exactly as human preference labels would be in standard RLHF: to train a preference model (reward model) using the Bradley-Terry loss, which is then used to fine-tune the language model via RL.

Key properties of RLAIF:

- **Scalability.** AI labelling can be done at orders of magnitude lower cost than human labelling. Millions of preference pairs can be generated automatically.
- **Consistency.** AI labels are more consistent than human labels, especially for subtle harms where annotators disagree. Consistency reduces noise in preference model training.
- **Transparency.** The principle used for each comparison is explicit in the prompt, making it possible to audit what criterion was applied to each pair.
- **Potential bias amplification.** The AI labeller inherits biases from its own pretraining. Constitutional AI attempts to mitigate this by using a diverse set of principles rather than a single labelling criterion.

---

## Intermediate

### Q5. How does Constitutional AI compare to standard RLHF in terms of human oversight?

**Answer.**

| Dimension | Standard RLHF | Constitutional AI (RLAIF) |
|---|---|---|
| Human involvement | Annotators compare many (prompt, response) pairs | Humans write the constitution; AI does the labelling |
| Scale | Limited by annotator throughput | Scales with AI compute |
| Transparency | Principles implicit in annotation guidelines | Principles explicit in the written constitution |
| Adjustability | Requires new annotation to change values | Change the constitution text |
| Annotator harm | Annotators must evaluate harmful content | AI evaluates harmful content; humans need not |
| Label quality | High variance; annotator disagreement common | More consistent but inherits AI biases |
| Verification | Hard to verify what the reward model learned | Can probe the AI labeller with test cases |

**The key tradeoff** is between human oversight and scalability. Standard RLHF maintains human oversight at every labelling step but is expensive and exposes annotators to harmful content. CAI/RLAIF concentrates human oversight at the constitution design stage — which is arguably higher leverage (one senior researcher writing principles vs. thousands of annotator decisions) but also a potential single point of failure.

**The risk of RLAIF** is that the AI labeller's judgements are only as good as the AI labeller itself. If the labeller has systematic biases (e.g., preferring longer responses, preferring responses that match common views even when wrong), those biases will be amplified through the preference model training. The constitution provides a mitigation but does not eliminate this risk.

---

### Q6. What are the design considerations for an effective constitution?

**Answer.**

Designing a constitution is a policy design problem. Key considerations:

**1. Coverage across harm categories.**
The constitution should cover all major categories of potential harm: physical harm, psychological harm, financial harm, privacy violations, discrimination, misinformation, and so on. Gaps in coverage will be exploited — the AI system will satisfy all stated principles while violating an unstated one.

**2. Diversity of ethical frameworks.**
Using principles from multiple ethical traditions (deontological rules like "do not assist in illegal activity," consequentialist principles like "choose the response with least harmful societal impact," virtue ethics like "choose the response that a wise and caring person would give") provides adversarial robustness. A model can circumvent a purely consequentialist constitution with sufficient cleverness, but covering multiple frameworks closes more loopholes.

**3. Positive vs negative framing.**
Principles stated positively ("choose the more helpful response") and negatively ("choose the less harmful response") train different aspects of behaviour. An exclusively negative constitution may produce a model that is safe but unhelpful. Anthropic's constitution explicitly includes helpfulness principles alongside harm avoidance.

**4. Specificity vs generality.**
Very general principles ("be ethical") are hard for an AI to apply consistently. Very specific principles ("do not provide instructions for synthesising Schedule I substances") are clear but leave gaps. Effective constitutions use layers: general principles for overall guidance, specific principles for known high-risk areas.

**5. Conflict resolution.**
Principles can conflict: being maximally helpful may conflict with being maximally cautious. The constitution should provide priority ordering or tie-breaking guidance. Anthropic uses a "thoughtful senior employee" heuristic as a meta-principle.

**6. Iterative refinement.**
Constitutions should be treated as living documents. Red-teaming the model against the current constitution reveals gaps, which can be addressed by adding or refining principles.

---

### Q7. How does CAI relate to scalable oversight and debate as alignment strategies?

**Answer.**

Constitutional AI is one approach in a broader class of "scalable oversight" methods — techniques designed to supervise increasingly capable AI systems even when human raters cannot evaluate the model's outputs directly.

**The scalable oversight problem:** as AI systems become more capable than humans at specific tasks (writing code, scientific reasoning), human annotators can no longer reliably judge which of two responses is better. Scalable oversight asks: how can we maintain oversight even then?

**CAI's approach to scalable oversight:** CAI uses AI-assisted evaluation to extend the reach of human oversight. Humans write principles (which they can verify are good principles), and the AI applies those principles at scale. This is scalable because principle application is computationally cheap once the model can apply principles well.

**Debate (Irving et al., 2018):** uses adversarial AI-vs-AI interactions where two AI systems argue for different answers and a human judge evaluates the argument. The hypothesis is that it is easier to judge an argument than to answer a question directly. Like CAI, this leverages AI to do the heavy lifting while keeping humans in the loop at the evaluation stage.

**Differences:**
- CAI is currently deployed in practice; debate is largely theoretical/experimental.
- CAI uses cooperative AI (the same model critiques itself using provided principles); debate uses adversarial AI (two models argue opposing positions).
- Debate has stronger theoretical backing for its oversight properties; CAI has stronger empirical backing.

Both methods assume that the capability gap between the AI evaluator and the AI being evaluated is not too large. If the model being evaluated is vastly more capable than the evaluating model, neither approach is guaranteed to work.

---

## Advanced

### Q8. What are the failure modes of RLAIF, and how does the choice of AI labeller affect the trained model?

**Answer.**

**Failure mode 1: Sycophancy amplification.**
If the AI labeller has a preference for agreeing with user statements, it will label responses that validate the user's views as "preferred." The preference model trained on these labels will reinforce this sycophancy. Because the feedback loop passes through multiple training stages, sycophancy can be amplified at each step.

**Failure mode 2: Length and formatting bias.**
Like human raters, AI labellers (especially those trained with RLHF themselves) tend to rate longer, more structured responses as better, independent of content quality. This can produce a preference model that rewards verbosity.

**Failure mode 3: Self-preference (model narcissism).**
When the AI labeller is the same model (or a close relative of) the model being trained, the labeller tends to prefer responses in its own "style." This can cause the trained model to overfit to stylistic features of the labeller rather than genuinely improving in quality.

**Failure mode 4: Constitutional principle gaming.**
The AI labeller may find responses that technically satisfy the stated constitutional principles while violating the intent. For example, a principle like "choose the less harmful response" can be satisfied by choosing a response that is only marginally less harmful, rather than the most helpful safe response.

**Mitigation strategies:**

- Use a different (ideally stronger) model as the AI labeller than the model being trained.
- Diverse principles in the constitution make it harder to satisfy all principles with a degenerate response.
- Periodic human audits of AI-generated preference labels catch systematic biases.
- Calibration checks: compare AI labeller agreement with human raters on a held-out set. If agreement is low, the AI labeller's labels are noisy.
- Mix AI-generated and human-generated preference labels; do not rely exclusively on RLAIF.

---

### Q9. How would you design a red-teaming pipeline to identify gaps in a constitution?

**Answer.**

Red-teaming a constitution means systematically probing the model for behaviours that violate the intended values but are not covered by any explicit principle.

**Step 1: Automated red-teaming.**
Train a red-teaming model specifically to generate prompts that elicit the target model's most harmful responses. This is itself a supervised fine-tuning task: start with a dataset of known harmful prompts and fine-tune a model to generate similar prompts. Run this red-teaming model against the CAI-trained model at scale and collect the most harmful responses.

**Step 2: Categorise violations.**
Cluster the harmful responses by violation type: physical harm instructions, psychological manipulation, privacy violations, misinformation, discrimination, etc. Each cluster represents a potential gap in the constitution.

**Step 3: Constitution gap analysis.**
For each violation cluster, check whether the existing constitution has a principle that would flag this violation. If not, draft a new principle. If yes, investigate why the principle failed to prevent the violation — either the AI labeller misapplied the principle, or the principle is underspecified.

**Step 4: Adversarial principle testing.**
For each new or modified principle, construct test cases that attempt to satisfy the letter of the principle while violating its intent. If the AI labeller endorses these adversarial responses, the principle is too weakly specified.

**Step 5: Human review.**
Have domain experts (legal, medical, ethics) review the constitution and red-teaming results for each harm category. Automated red-teaming is efficient but may miss culturally specific or domain-specific harms that require expert knowledge.

**Step 6: Iterate.**
Retrain the CAI model with the updated constitution and repeat the red-teaming process. Track the rate of policy violations across training iterations. A constitution that is not improving the model on the red-team test set needs further refinement.

---

### Q10. Compare Constitutional AI with RLHF and DPO. When would you choose each approach?

**Answer.**

| Criterion | RLHF (PPO) | DPO | Constitutional AI |
|---|---|---|---|
| Human feedback required | Many pairwise comparisons | Many pairwise comparisons | Mostly for constitution design |
| Annotation cost | High | High | Low (at scale) |
| Transparency of values | Low (implicit in RM) | Low (implicit in RM) | High (explicit constitution) |
| Adjustability | Retrain RM | Retrain RM | Edit constitution text |
| Scalability | Limited by annotators | Limited by annotators | Scales with compute |
| Handling novel harms | Requires new annotation | Requires new annotation | Add a principle |
| Theoretical grounding | Strong (RL theory) | Strong (closed-form derivation) | Moderate (empirical) |
| Production maturity | High | High | Moderate |

**Choose RLHF when:**
- You have access to a large, high-quality preference dataset with reliable human labels.
- You need maximum alignment quality and have the compute budget for PPO.
- The task has a relatively clear preference signal (e.g., factual accuracy, instruction following).

**Choose DPO when:**
- You have a good preference dataset but want stability and lower compute requirements.
- You are iterating quickly and need reproducible training runs.
- The preference signal can be captured from an existing dataset without online data collection.

**Choose Constitutional AI / RLAIF when:**
- Annotation budget is limited but compute budget is available.
- You need transparent, auditable, adjustable values.
- You are building a system where principles may change frequently (e.g., adapting to different regulatory environments or use cases).
- You need to avoid exposing human annotators to harmful content at scale.

In practice, these approaches are not mutually exclusive. Anthropic uses elements of all three: SFT on high-quality human demonstrations, CAI for scalable preference labelling, and RL for the final policy optimisation step.
