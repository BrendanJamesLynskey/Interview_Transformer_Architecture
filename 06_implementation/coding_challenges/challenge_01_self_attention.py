"""
Challenge 01: Scaled Dot-Product Attention from Scratch
=========================================================
Implement scaled dot-product attention in PyTorch, including:
  - Basic attention (no masking)
  - Causal (autoregressive) masking
  - Padding masking

Then test your implementation against torch.nn.functional.scaled_dot_product_attention.

Learning objectives:
  1. Understand the attention formula and why each step exists.
  2. Implement numerically stable softmax with masking.
  3. Handle both causal and padding masks correctly.
  4. Debug shape mismatches common in attention implementations.
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Args:
        query:     (..., T_q, d_k)  Query tensor. Any leading batch dims are supported.
        key:       (..., T_k, d_k)  Key tensor.
        value:     (..., T_k, d_v)  Value tensor.
        attn_mask: Optional boolean mask of shape broadcastable to (..., T_q, T_k).
                   True means "mask out this position" (assign -inf before softmax).
        dropout_p: Dropout probability applied to attention weights.
        is_causal: If True, applies a causal (upper-triangular) mask automatically.
                   attn_mask is ignored when is_causal=True.

    Returns:
        output:       (..., T_q, d_v)  Weighted value vectors.
        attn_weights: (..., T_q, T_k)  Attention probabilities (after softmax).
    """
    # --- Step 1: Extract dimensions ---
    # d_k is the key/query dimension. We scale by its square root.
    d_k = query.size(-1)
    T_q = query.size(-2)
    T_k = key.size(-2)

    # --- Step 2: Compute raw attention scores ---
    # Shape: (..., T_q, T_k)
    # Each query position produces a score against every key position.
    # We scale by 1/sqrt(d_k) to prevent large-magnitude logits that
    # would make softmax very sharp and gradients near zero.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # --- Step 3: Apply masking ---
    # Masking must happen BEFORE softmax. Masked positions receive -inf so that
    # exp(-inf) = 0 and they contribute zero weight after normalisation.
    if is_causal:
        # Build an upper-triangular boolean mask: True = should be masked out.
        # The diagonal (i == j) is NOT masked: each position can attend to itself.
        # We create it on the same device as scores.
        causal_mask = torch.ones(T_q, T_k, dtype=torch.bool, device=query.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)  # True in upper triangle
        scores = scores.masked_fill(causal_mask, float('-inf'))
    elif attn_mask is not None:
        # User-supplied boolean mask (True = mask out).
        scores = scores.masked_fill(attn_mask, float('-inf'))

    # --- Step 4: Numerically stable softmax ---
    # Softmax is applied over the key dimension (dim=-1) so that for each
    # query position, the attention weights over all key positions sum to 1.
    #
    # We cast to float32 for the softmax computation to avoid fp16 overflow,
    # then cast back to the original dtype. This is what production kernels do.
    original_dtype = scores.dtype
    attn_weights = F.softmax(scores.float(), dim=-1).to(original_dtype)

    # After softmax, any position that received -inf will have weight exactly 0.
    # We explicitly zero these out to prevent NaN propagation (0 * -inf = NaN).
    if is_causal or attn_mask is not None:
        # Replace any NaN values that may arise from 0 * -inf in edge cases
        # (e.g., a row where ALL keys are masked becomes 0/0 = NaN).
        attn_weights = attn_weights.nan_to_num(0.0)

    # --- Step 5: Optional dropout on attention weights ---
    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # --- Step 6: Weighted sum of values ---
    # Shape: (..., T_q, d_v)
    # Each query position gets a weighted combination of value vectors.
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


# ---------------------------------------------------------------------------
# Utility: build a causal mask as a standalone function
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns a boolean mask of shape (seq_len, seq_len).
    mask[i, j] = True means position i should NOT attend to position j.
    True for all j > i (upper triangle, diagonal excluded).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                      diagonal=1)
    return mask


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Given a 1-D tensor of actual sequence lengths, returns a boolean padding mask
    of shape (batch, 1, 1, max_len) where True means "this key position is padding."

    Suitable for broadcasting against attention score tensors of shape (B, H, T_q, T_k).

    Args:
        lengths: (B,) tensor of actual sequence lengths.
        max_len: The padded sequence length.

    Returns:
        (B, 1, 1, max_len) boolean mask.
    """
    # positions: (1, max_len) tensor of position indices 0..max_len-1
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    # lengths:   (B, 1) for broadcasting
    lengths = lengths.unsqueeze(1)
    # True where position >= length (i.e., padding)
    mask = positions >= lengths                            # (B, max_len)
    return mask.unsqueeze(1).unsqueeze(1)                 # (B, 1, 1, max_len)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_attention():
    """Test that basic (no-mask) attention matches F.scaled_dot_product_attention."""
    print("Test 1: Basic attention (no mask)")
    B, T, d_k, d_v = 2, 10, 32, 32

    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_v)

    out_custom, weights_custom = scaled_dot_product_attention(Q, K, V)

    # PyTorch reference (no mask, no dropout)
    out_ref = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)

    assert out_custom.shape == (B, T, d_v), f"Wrong output shape: {out_custom.shape}"
    assert weights_custom.shape == (B, T, T), f"Wrong weights shape: {weights_custom.shape}"

    max_diff = (out_custom - out_ref).abs().max().item()
    print(f"  Max absolute difference from PyTorch reference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff}"

    # Verify weights sum to 1 over keys
    weight_sums = weights_custom.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        "Attention weights do not sum to 1"
    print("  PASSED")


def test_causal_mask():
    """Test that causal masking prevents future attention."""
    print("Test 2: Causal masking")
    B, T, d_k = 1, 8, 16

    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    out_custom, weights_custom = scaled_dot_product_attention(Q, K, V, is_causal=True)
    out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    # Verify that future positions have zero attention weight
    for i in range(T):
        for j in range(i + 1, T):
            w = weights_custom[0, i, j].item()
            assert abs(w) < 1e-6, \
                f"Non-zero attention weight at position ({i},{j}): {w}"

    max_diff = (out_custom - out_ref).abs().max().item()
    print(f"  Max absolute difference from PyTorch reference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff}"

    # Verify attention matrix is lower triangular
    upper = torch.triu(weights_custom[0], diagonal=1)
    assert upper.abs().max() < 1e-6, "Causal mask not applied correctly"
    print("  PASSED")


def test_padding_mask():
    """Test that padding positions receive zero attention weight."""
    print("Test 3: Padding mask")
    B, T, d_k = 3, 10, 16

    # Sequences of lengths 10, 7, 4 (rest is padding)
    lengths = torch.tensor([10, 7, 4])
    pad_mask = make_padding_mask(lengths, T)  # (B, 1, 1, T)

    Q = torch.randn(B, T, d_k)
    K = torch.randn(B, T, d_k)
    V = torch.randn(B, T, d_k)

    # Expand Q/K/V to (B, 1, T, d_k) for multi-head compatibility
    Q_4d = Q.unsqueeze(1)
    K_4d = K.unsqueeze(1)
    V_4d = V.unsqueeze(1)

    out_custom, weights_custom = scaled_dot_product_attention(
        Q_4d, K_4d, V_4d, attn_mask=pad_mask
    )

    # Check that padded key positions have zero attention weight
    for b, length in enumerate(lengths.tolist()):
        padded_weights = weights_custom[b, 0, :, length:]  # (T_q, n_padded)
        assert padded_weights.abs().max() < 1e-6, \
            f"Non-zero attention to padding in batch {b}: {padded_weights.abs().max()}"

    print(f"  Lengths: {lengths.tolist()}")
    print(f"  All padded positions have zero attention weight")
    print("  PASSED")


def test_gradient_flow():
    """Test that gradients flow to Q, K, and V."""
    print("Test 4: Gradient flow")
    B, T, d_k = 2, 6, 16

    Q = torch.randn(B, T, d_k, requires_grad=True)
    K = torch.randn(B, T, d_k, requires_grad=True)
    V = torch.randn(B, T, d_k, requires_grad=True)

    out, _ = scaled_dot_product_attention(Q, K, V)
    loss = out.sum()
    loss.backward()

    assert Q.grad is not None and Q.grad.abs().max() > 0, "No gradient for Q"
    assert K.grad is not None and K.grad.abs().max() > 0, "No gradient for K"
    assert V.grad is not None and V.grad.abs().max() > 0, "No gradient for V"

    print(f"  Q grad norm: {Q.grad.norm():.4f}")
    print(f"  K grad norm: {K.grad.norm():.4f}")
    print(f"  V grad norm: {V.grad.norm():.4f}")
    print("  PASSED")


def test_different_query_key_lengths():
    """Test cross-attention scenario where T_q != T_k."""
    print("Test 5: Cross-attention (T_q != T_k)")
    B, T_q, T_k, d_k, d_v = 2, 5, 12, 16, 32

    Q = torch.randn(B, T_q, d_k)
    K = torch.randn(B, T_k, d_k)
    V = torch.randn(B, T_k, d_v)

    out, weights = scaled_dot_product_attention(Q, K, V)

    assert out.shape == (B, T_q, d_v), f"Wrong output shape: {out.shape}"
    assert weights.shape == (B, T_q, T_k), f"Wrong weights shape: {weights.shape}"

    out_ref = F.scaled_dot_product_attention(Q, K, V)
    max_diff = (out - out_ref).abs().max().item()
    assert max_diff < 1e-4, f"Cross-attention outputs differ: {max_diff}"
    print(f"  Q shape: {Q.shape}, K shape: {K.shape}, output shape: {out.shape}")
    print("  PASSED")


def test_multi_head_shape():
    """Test with 4D tensors as used in multi-head attention (B, H, T, d_k)."""
    print("Test 6: 4D tensors (batch x heads x seq x dim)")
    B, H, T, d_k = 2, 8, 16, 64

    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)
    V = torch.randn(B, H, T, d_k)

    out, weights = scaled_dot_product_attention(Q, K, V, is_causal=True)
    out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    assert out.shape == (B, H, T, d_k)
    max_diff = (out - out_ref).abs().max().item()
    assert max_diff < 1e-4, f"4D attention outputs differ: {max_diff}"
    print(f"  Input shape: {Q.shape}, output shape: {out.shape}")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge 01: Scaled Dot-Product Attention")
    print("=" * 60)
    print()

    test_basic_attention()
    print()
    test_causal_mask()
    print()
    test_padding_mask()
    print()
    test_gradient_flow()
    print()
    test_different_query_key_lengths()
    print()
    test_multi_head_shape()
    print()
    print("All tests passed.")
