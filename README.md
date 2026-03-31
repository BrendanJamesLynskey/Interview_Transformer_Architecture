# Transformer Architecture — Interview Preparation

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive interview preparation resource covering the Transformer architecture from the foundational "Attention Is All You Need" paper through modern large language model architectures. This repository provides deep mathematical understanding, worked problem solutions, and coding implementations for technical interviews.

## Overview

This repository is designed for engineers preparing for technical interviews on deep learning and language models. It covers:

- Core attention mechanisms and the original Transformer architecture
- Training methodologies including pretraining, tokenization, and scaling laws
- Modern architectural innovations (RoPE, GQA, MoE, Flash Attention)
- Inference optimization techniques and inference-time engineering
- Alignment techniques (SFT, RLHF, DPO, Constitutional AI)
- Practical implementation from scratch
- Quiz materials for self-assessment

## Table of Contents

1. [01 Foundations](#01-foundations)
2. [02 Training](#02-training)
3. [03 Modern Architectures](#03-modern-architectures)
4. [04 Inference Optimisation](#04-inference-optimisation)
5. [05 Alignment](#05-alignment)
6. [06 Implementation](#06-implementation)
7. [07 Quizzes](#07-quizzes)

## 01 Foundations

Core concepts from "Attention Is All You Need" and the fundamentals of Transformer architecture.

- `attention_mechanism.md` — Conceptual overview of attention and why it matters
- `scaled_dot_product_attention.md` — Mathematical formulation of scaled dot-product attention
- `multi_head_attention.md` — Multiple attention heads and their interpretation
- `positional_encoding.md` — Sinusoidal positional encodings and alternatives
- `encoder_decoder_structure.md` — Encoder and decoder stack architecture
- `worked_problems/` — Detailed solutions to foundational problems
  - `problem_01_attention_score_calculation.md` — Computing attention scores
  - `problem_02_parameter_counting.md` — Counting parameters in attention layers
  - `problem_03_positional_encoding_derivation.md` — Deriving positional encodings

## 02 Training

Training methodologies and techniques for large-scale model development.

- `pretraining_objectives.md` — Language modeling objectives and variants
- `tokenisation_bpe_sentencepiece.md` — Subword tokenization algorithms
- `learning_rate_schedules.md` — Scheduling strategies and warmup
- `scaling_laws.md` — Chinchilla and scaling law fundamentals
- `training_stability.md` — Gradient clipping, loss spikes, and stabilization
- `worked_problems/` — Training-focused problem solutions
  - `problem_01_loss_calculation.md` — Computing cross-entropy loss
  - `problem_02_compute_optimal_scaling.md` — Optimal model/data scaling
  - `problem_03_batch_size_analysis.md` — Batch size and learning rate relationships

## 03 Modern Architectures

Recent architectural innovations and optimizations in transformer-based LLMs.

- `decoder_only_llms.md` — Why decoder-only models dominate modern LLMs
- `rope_and_alibi.md` — Rotary position embeddings and ALiBi
- `grouped_query_attention.md` — Grouped query attention for efficiency
- `mixture_of_experts.md` — MoE routing and training considerations
- `flash_attention.md` — IO-aware attention for faster computation
- `rmsnorm_and_pre_norm.md` — RMSNorm and pre-normalization architectures
- `worked_problems/` — Modern architecture problem solutions
  - `problem_01_rope_rotation_matrices.md` — Applying RoPE rotations
  - `problem_02_moe_routing.md` — Computing MoE routing decisions
  - `problem_03_flash_attention_tiling.md` — Tiling strategies in Flash Attention

## 04 Inference Optimisation

Techniques for efficient inference and inference-time engineering.

- `kv_cache_mechanism.md` — KV cache mechanics and memory calculations
- `quantisation_ptq_and_qlora.md` — Post-training quantization and QLoRA
- `pruning_and_distillation.md` — Model pruning and knowledge distillation
- `speculative_decoding.md` — Speedup techniques for autoregressive generation
- `continuous_batching.md` — Paged attention and continuous batching systems
- `worked_problems/` — Inference optimization problem solutions
  - `problem_01_kv_cache_memory.md` — Calculating KV cache memory usage
  - `problem_02_quantisation_error.md` — Quantization error analysis
  - `problem_03_speculative_acceptance_rate.md` — Acceptance rates in speculative decoding

## 05 Alignment

Post-training alignment techniques and preference optimization.

- `sft_supervised_fine_tuning.md` — Supervised fine-tuning for instruction following
- `rlhf_and_reward_models.md` — Reinforcement learning from human feedback
- `dpo_direct_preference_optimisation.md` — Direct preference optimization
- `constitutional_ai.md` — Constitutional AI and principled alignment
- `worked_problems/` — Alignment problem solutions
  - `problem_01_rlhf_loss_derivation.md` — Deriving RLHF objectives
  - `problem_02_dpo_vs_rlhf.md` — Comparing DPO and RLHF
  - `problem_03_reward_hacking.md` — Identifying and preventing reward hacking

## 06 Implementation

Practical implementation of Transformer components from scratch.

- `implementing_attention_from_scratch.md` — Step-by-step guide to building attention
- `coding_challenges/` — Implementation exercises
  - `challenge_01_self_attention.py` — Self-attention in PyTorch
  - `challenge_02_multi_head_attention.py` — Multi-head attention layer
  - `challenge_03_transformer_block.py` — Complete transformer block
  - `challenge_04_positional_encoding.py` — Positional encoding implementations
  - `challenge_05_bpe_tokeniser.py` — BPE tokenizer from scratch
  - `challenge_06_kv_cache.py` — Efficient KV cache implementation

## 07 Quizzes

Self-assessment quizzes covering all major topics.

- `quiz_attention_fundamentals.md` — Foundation-level quiz
- `quiz_training.md` — Training methodology quiz
- `quiz_modern_architectures.md` — Modern architecture features quiz
- `quiz_inference.md` — Inference optimization quiz
- `quiz_alignment.md` — Alignment techniques quiz

## How to Use

This repository is structured as an interview preparation guide. Suggested approaches:

1. **Foundational learning**: Start with section 01 (Foundations) to understand core concepts from the original Transformer paper. Work through the problems in each subsection.

2. **Deep dives**: For each topic, read the conceptual document first, then attempt the worked problems before reviewing solutions. Use these problems to practice explaining concepts clearly.

3. **Modern context**: Section 03 covers innovations that have become standard in production LLMs. Understanding these is essential for current interviews.

4. **Practical skills**: Section 06 includes coding challenges. Implement these from scratch to understand engineering trade-offs.

5. **Self-assessment**: Use the quizzes in section 07 to identify knowledge gaps and review weak areas.

6. **Interview simulation**: Practice explaining concepts from each section in 2-3 minute blocks, as if presenting to an interviewer.

## Related Repositories

- **[LLM_Transformer_Decoder_guide](https://github.com/BrendanJamesLynskey/LLM_Transformer_Decoder_guide)** — A comprehensive tutorial walkthrough of decoder-only transformer architectures. While this repository focuses on interview Q&A with mathematical derivations and problem-solving, that repository provides detailed implementation guides and architectural walkthroughs.

## Contributing

Contributions are welcome. Please consider:

- Verify technical accuracy, especially mathematical derivations
- Add new worked problems with complete solutions
- Expand quiz materials with diverse question types
- Improve clarity in explanations and derivations
- Add implementation examples for new techniques

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

**Last updated**: 2026-03-31
