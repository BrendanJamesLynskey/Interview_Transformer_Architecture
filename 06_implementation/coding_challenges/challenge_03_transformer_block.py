"""
Challenge 03: Transformer Decoder Block from Scratch
=====================================================
Implement a full Transformer decoder block as used in modern LLMs (Llama-style):
  - RMSNorm (pre-norm) instead of LayerNorm
  - Multi-head self-attention with causal masking (or GQA)
  - SwiGLU feed-forward network
  - Residual connections (post-add, not post-norm)

This block architecture matches Llama 2/3, Mistral, and other modern decoder-only models.

Architecture (per block):
    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

where FFN uses SwiGLU:
    FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

Learning objectives:
  1. Understand pre-norm vs post-norm and why pre-norm is preferred in large models.
  2. Implement RMSNorm from scratch.
  3. Implement SwiGLU and understand why it outperforms GELU feed-forward.
  4. Understand the role of residual connections in deep networks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

    RMSNorm normalises by the RMS of the input rather than the mean and variance:

        RMSNorm(x) = x / RMS(x) * gamma

    where RMS(x) = sqrt(mean(x^2) + eps) and gamma is a learned scale parameter.

    Compared to LayerNorm:
      - No mean subtraction (re-centering): faster, similar performance.
      - No beta (bias) parameter: fewer parameters, no benefit observed empirically.
      - Used in Llama, Gemma, Mistral, PaLM, and most modern LLMs.

    Args:
        d_model: Feature dimension to normalise over.
        eps:     Small value added to denominator for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable scale parameter; shape (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d_model) input tensor.
        Returns:
            (..., d_model) normalised tensor.
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        # Keep dims for broadcasting; compute in fp32 for numerical stability.
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        # Normalise and scale
        return (x.float() / rms).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer, 2020).

    The standard Transformer FFN is:
        FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2

    SwiGLU uses a gated mechanism:
        FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    The SiLU activation (also called Swish) is: SiLU(x) = x * sigmoid(x).
    The elementwise product creates a "gate" that controls information flow.

    SwiGLU outperforms standard GELU-FFN at the same parameter count. To keep
    the parameter count comparable to a standard FFN with hidden dimension 4*d,
    the intermediate dimension is scaled down:
        intermediate_dim = int(2/3 * 4 * d_model) rounded to a multiple of 256.

    This is the exact formula used in Llama.

    Args:
        d_model:          Input/output dimension.
        intermediate_dim: Hidden dimension. If None, computed as (8/3)*d_model.
        bias:             Whether to include bias terms.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_dim: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if intermediate_dim is None:
            # Llama formula: (8/3 * d_model) rounded to nearest multiple of 256
            intermediate_dim = int(8 * d_model / 3)
            intermediate_dim = (intermediate_dim + 255) // 256 * 256

        self.d_model = d_model
        self.intermediate_dim = intermediate_dim

        # Gate projection: x -> gate (will be passed through SiLU)
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        # Up projection: x -> hidden (element-wise multiplied by gate)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        # Down projection: hidden -> output
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        # SwiGLU: SiLU(gate) * up, then project down
        # F.silu implements x * sigmoid(x)
        gate = F.silu(self.gate_proj(x))   # (B, T, intermediate_dim)
        up = self.up_proj(x)               # (B, T, intermediate_dim)
        hidden = gate * up                 # Elementwise gating
        return self.down_proj(hidden)      # (B, T, d_model)


# ---------------------------------------------------------------------------
# Multi-Head Attention (from Challenge 02, simplified version)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.
    Simplified version of Challenge 02's MultiHeadAttention, self-attention only.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Fused QKV projection for efficiency: one matmul instead of three
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, d_model = x.shape

        # Fused QKV: one matmul, then split
        qkv = self.qkv_proj(x)                         # (B, T, 3*d_model)
        Q, K, V = qkv.split(d_model, dim=-1)           # Each: (B, T, d_model)

        # Reshape to (B, num_heads, T, d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Causal attention (is_causal=True handles the mask internally)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, num_heads, T, d_k)

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer Decoder Block
# ---------------------------------------------------------------------------

class TransformerDecoderBlock(nn.Module):
    """
    A single Transformer decoder block (Llama-style architecture).

    Uses pre-norm: normalisation is applied BEFORE each sub-layer, not after.
    This is opposite to the original Transformer (post-norm) but is empirically
    more stable at large scale and allows training without warm-up schedules.

    Architecture:
        x = x + Attention(RMSNorm(x))   # Self-attention sub-layer
        x = x + FFN(RMSNorm(x))         # Feed-forward sub-layer

    Note: the residual connections bypass the sub-layers, not the norms.
    This preserves a clear gradient highway from output to input.

    Args:
        d_model:          Model dimension.
        num_heads:        Number of attention heads.
        intermediate_dim: FFN hidden dimension. None = Llama default (8/3 * d_model).
        dropout:          Dropout on attention weights.
        norm_eps:         Epsilon for RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        intermediate_dim: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        # Pre-attention norm
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        # Self-attention
        self.attn = CausalSelfAttention(d_model, num_heads, dropout=dropout)

        # Pre-FFN norm
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        # Feed-forward
        self.ffn = SwiGLUFFN(d_model, intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        # Pre-norm self-attention + residual
        x = x + self.attn(self.norm1(x))

        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x


# ---------------------------------------------------------------------------
# Minimal Decoder-Only LM using stacked blocks
# ---------------------------------------------------------------------------

class TinyDecoderLM(nn.Module):
    """
    A minimal decoder-only language model for testing the block implementation.

    Components:
      - Token embedding table
      - Stack of TransformerDecoderBlocks
      - Final RMSNorm
      - Language modelling head (linear to vocabulary)

    This matches the high-level structure of Llama/GPT models.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.norm_final = RMSNorm(d_model)
        # Weight tying: the output projection shares weights with the embedding.
        # This is used in GPT-2, Llama, and most modern LLMs.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) integer token IDs.
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.token_emb(input_ids)    # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)           # (B, T, d_model)
        logits = self.lm_head(x)         # (B, T, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rmsnorm():
    """Verify RMSNorm normalises correctly and has learnable weights."""
    print("Test 1: RMSNorm")
    B, T, d = 2, 10, 64
    norm = RMSNorm(d)
    x = torch.randn(B, T, d)
    out = norm(x)

    assert out.shape == (B, T, d), f"Wrong shape: {out.shape}"

    # With gamma=1, output should have RMS ~= 1 per vector
    rms = out.pow(2).mean(dim=-1).sqrt()
    print(f"  Output RMS (should be ~1.0): mean={rms.mean():.4f}, std={rms.std():.4f}")
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), \
        "RMSNorm output does not have unit RMS"

    # Scaling: doubling gamma should double output
    norm.weight.data.fill_(2.0)
    out2 = norm(x)
    assert torch.allclose(out2, 2 * out, atol=1e-5), "Weight scaling not working"
    print("  RMS values are ~1.0. Weight scaling works.")
    print("  PASSED")


def test_swiglu_ffn():
    """Test SwiGLU FFN shape and gradient flow."""
    print("Test 2: SwiGLU Feed-Forward Network")
    B, T, d = 2, 8, 64
    ffn = SwiGLUFFN(d)

    x = torch.randn(B, T, d)
    out = ffn(x)

    assert out.shape == (B, T, d), f"Wrong output shape: {out.shape}"
    print(f"  d_model={d}, intermediate_dim={ffn.intermediate_dim}")
    print(f"  Input shape: {x.shape}, output shape: {out.shape}")

    # Gradient flow
    out.sum().backward()
    for name, param in ffn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    print("  Gradients flow to all FFN parameters")
    print("  PASSED")


def test_decoder_block():
    """Test the full decoder block: shape and residual connections."""
    print("Test 3: TransformerDecoderBlock")
    B, T, d, H = 2, 12, 64, 4
    block = TransformerDecoderBlock(d, H)
    x = torch.randn(B, T, d)
    out = block(x)

    assert out.shape == (B, T, d), f"Wrong output shape: {out.shape}"
    print(f"  Input shape: {x.shape}, output shape: {out.shape}")

    # Verify residual connections: output should not be identical to input
    # (unless weights happen to produce zero updates)
    assert not torch.allclose(out, x), "Output is identical to input (residual broken?)"
    print("  Output differs from input (residual connections present)")

    # Gradient flow to input
    x.requires_grad_(True)
    out = block(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().max() > 0, "No gradient to input"
    print(f"  Input gradient norm: {x.grad.norm():.4f}")
    print("  PASSED")


def test_causal_masking_in_block():
    """
    Verify that the block respects causality: changing a later token should not
    affect the output at earlier positions.
    """
    print("Test 4: Causal masking in decoder block")
    B, T, d, H = 1, 8, 32, 4
    block = TransformerDecoderBlock(d, H)
    block.eval()

    x1 = torch.randn(B, T, d)
    x2 = x1.clone()
    # Modify only the last token
    x2[:, -1, :] = torch.randn(d)

    with torch.no_grad():
        out1 = block(x1)
        out2 = block(x2)

    # Positions 0..T-2 should be identical; only position T-1 can differ.
    diff_early = (out1[:, :-1, :] - out2[:, :-1, :]).abs().max().item()
    diff_last = (out1[:, -1, :] - out2[:, -1, :]).abs().max().item()

    print(f"  Diff at positions 0..{T-2} (should be ~0): {diff_early:.2e}")
    print(f"  Diff at position {T-1} (should be >0): {diff_last:.2e}")
    assert diff_early < 1e-5, f"Earlier positions affected by last-token change: {diff_early}"
    assert diff_last > 1e-5, "Last position unaffected by token change (unexpected)"
    print("  PASSED")


def test_tiny_lm():
    """Test the full TinyDecoderLM: forward pass and cross-entropy loss."""
    print("Test 5: TinyDecoderLM — forward pass and loss")
    vocab_size = 1000
    d_model, num_heads, num_layers = 64, 4, 2
    B, T = 2, 16

    model = TinyDecoderLM(vocab_size, d_model, num_heads, num_layers)

    input_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)

    assert logits.shape == (B, T, vocab_size), \
        f"Wrong logits shape: {logits.shape}"
    print(f"  Input: {input_ids.shape}, logits: {logits.shape}")

    # Compute cross-entropy loss for language modelling
    # Predict token t+1 from token t: shift inputs and targets by 1.
    targets = input_ids[:, 1:]     # (B, T-1)
    logits_shifted = logits[:, :-1, :]  # (B, T-1, vocab)
    loss = F.cross_entropy(
        logits_shifted.reshape(-1, vocab_size),
        targets.reshape(-1),
    )
    print(f"  Cross-entropy loss (random init, should be ~log({vocab_size})={math.log(vocab_size):.2f}): {loss.item():.4f}")

    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print("  PASSED")


def test_pre_norm_vs_post_norm():
    """
    Demonstrate why pre-norm is preferred: gradient norms are more stable.
    Compare gradient norms at different layers for pre-norm vs post-norm stacks.
    """
    print("Test 6: Pre-norm gradient stability (informational)")

    class PostNormBlock(nn.Module):
        """Classic post-norm Transformer block (original paper style)."""
        def __init__(self, d: int, H: int):
            super().__init__()
            self.attn = CausalSelfAttention(d, H)
            self.ffn = SwiGLUFFN(d)
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)

        def forward(self, x):
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.ffn(x))
            return x

    d, H, B, T = 64, 4, 2, 10
    n_layers = 8

    pre_norm_model = nn.Sequential(*[TransformerDecoderBlock(d, H) for _ in range(n_layers)])
    post_norm_model = nn.Sequential(*[PostNormBlock(d, H) for _ in range(n_layers)])

    x = torch.randn(B, T, d, requires_grad=True)

    for name, model in [("Pre-norm", pre_norm_model), ("Post-norm", post_norm_model)]:
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Collect gradient norms for first-layer parameters
        first_layer_grad_norms = []
        for p in list(model[0].parameters()):
            if p.grad is not None:
                first_layer_grad_norms.append(p.grad.norm().item())

        avg_first = sum(first_layer_grad_norms) / len(first_layer_grad_norms) if first_layer_grad_norms else 0

        # Collect gradient norms for last-layer parameters
        last_layer_grad_norms = []
        for p in list(model[-1].parameters()):
            if p.grad is not None:
                last_layer_grad_norms.append(p.grad.norm().item())

        avg_last = sum(last_layer_grad_norms) / len(last_layer_grad_norms) if last_layer_grad_norms else 0
        ratio = avg_first / (avg_last + 1e-8)

        print(f"  {name}: first layer avg grad={avg_first:.4f}, last layer avg grad={avg_last:.4f}, ratio={ratio:.2f}")
        # Reset grads for next iteration
        x.grad = None
        for p in model.parameters():
            p.grad = None

    print("  (Pre-norm should have ratio closer to 1.0 — more uniform gradients)")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge 03: Transformer Decoder Block")
    print("=" * 60)
    print()

    test_rmsnorm()
    print()
    test_swiglu_ffn()
    print()
    test_decoder_block()
    print()
    test_causal_masking_in_block()
    print()
    test_tiny_lm()
    print()
    test_pre_norm_vs_post_norm()
    print()
    print("All tests passed.")
