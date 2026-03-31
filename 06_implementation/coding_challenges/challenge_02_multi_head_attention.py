"""
Challenge 02: Multi-Head Attention from Scratch
================================================
Implement multi-head attention (MHA) with:
  - Separate W_Q, W_K, W_V projection matrices
  - Head splitting and concatenation
  - Output projection W_O
  - Causal masking support

Then test against torch.nn.MultiheadAttention.

Learning objectives:
  1. Understand how heads are split from a single projection (no separate per-head matrices).
  2. Handle the shape transformations correctly, especially transpose + reshape.
  3. Understand why the output projection is necessary.
  4. See how GQA (Grouped Query Attention) can be implemented as a generalisation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head self/cross-attention as described in Vaswani et al. (2017).

    Uses a single linear projection for all heads in each of Q, K, V, then
    splits the projected dimension into h heads. This is equivalent to h
    separate small projections but more efficient as a single batched matmul.

    Args:
        d_model: Total model dimension. Must be divisible by num_heads.
        num_heads: Number of attention heads.
        dropout: Dropout probability applied to attention weights.
        bias: Whether to include bias terms in Q/K/V/O projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # Dimension per head
        self.dropout = dropout

        # Single projection for all heads combined.
        # W_Q maps d_model -> d_model (which will be split into num_heads x d_k)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection recombines the concatenated head outputs.
        # Input: d_model (= num_heads * d_k), output: d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier uniform initialisation, matching PyTorch's MultiheadAttention."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            query:            (B, T_q, d_model)
            key:              (B, T_k, d_model)
            value:            (B, T_k, d_model)
            key_padding_mask: (B, T_k) boolean, True = padding position.
            is_causal:        Apply causal (autoregressive) mask.
            need_weights:     If True, also return attention weights.

        Returns:
            output:       (B, T_q, d_model)
            attn_weights: (B, num_heads, T_q, T_k) or None
        """
        B, T_q, _ = query.shape
        T_k = key.size(1)

        # --- Step 1: Project Q, K, V ---
        # Each projection maps (B, T, d_model) -> (B, T, d_model)
        Q = self.q_proj(query)   # (B, T_q, d_model)
        K = self.k_proj(key)     # (B, T_k, d_model)
        V = self.v_proj(value)   # (B, T_k, d_model)

        # --- Step 2: Split into heads ---
        # Reshape d_model -> (num_heads, d_k), then move heads dimension before T.
        # (B, T, d_model) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        Q = Q.view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        # Shapes are now (B, num_heads, T, d_k)

        # --- Step 3: Build combined attention mask ---
        # F.scaled_dot_product_attention accepts an additive float mask (not boolean).
        # We build a combined mask from causal + padding requirements.
        attn_mask = None

        if is_causal:
            # Upper-triangular boolean mask: True = mask out (future positions).
            # Shape: (T_q, T_k) — broadcasts across B and num_heads.
            causal = torch.triu(
                torch.ones(T_q, T_k, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            # Convert bool mask to additive float mask: True -> -inf, False -> 0.
            attn_mask = torch.zeros(T_q, T_k, device=query.device, dtype=Q.dtype)
            attn_mask = attn_mask.masked_fill(causal, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_k), True = padding.
            # We need shape (B, 1, 1, T_k) to broadcast over heads and query positions.
            pad_float = torch.zeros(
                B, 1, 1, T_k, device=query.device, dtype=Q.dtype
            )
            pad_float = pad_float.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf')
            )
            attn_mask = pad_float if attn_mask is None else attn_mask + pad_float

        # --- Step 4: Scaled dot-product attention ---
        # PyTorch 2.0+ dispatches to FlashAttention when possible.
        # We use is_causal=False here because we already built the causal mask above.
        if need_weights:
            # Manual computation to return attention weights.
            # Scale: 1/sqrt(d_k)
            scale = 1.0 / math.sqrt(self.d_k)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, T_q, T_k)
            if attn_mask is not None:
                scores = scores + attn_mask
            weights = F.softmax(scores.float(), dim=-1).to(Q.dtype)
            weights = weights.nan_to_num(0.0)
            if self.dropout > 0.0 and self.training:
                weights = F.dropout(weights, p=self.dropout)
            attn_out = torch.matmul(weights, V)  # (B, H, T_q, d_k)
        else:
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (B, num_heads, T_q, d_k)
            weights = None

        # --- Step 5: Concatenate heads and project output ---
        # (B, num_heads, T_q, d_k) -> (B, T_q, num_heads, d_k) -> (B, T_q, d_model)
        # Note: transpose creates a non-contiguous tensor; use .contiguous() before .view()
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # Final linear projection: (B, T_q, d_model) -> (B, T_q, d_model)
        output = self.out_proj(attn_out)

        return output, weights


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA) — used in Llama 2/3, Mistral, Gemma
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) as used in Llama-2 70B, Mistral, etc.

    In GQA:
      - There are `num_heads` query heads (each with its own projection).
      - There are `num_kv_heads` key/value heads (fewer than query heads).
      - Each KV head is shared by (num_heads / num_kv_heads) query heads.

    Special cases:
      - GQA with num_kv_heads == num_heads: standard MHA
      - GQA with num_kv_heads == 1: Multi-Query Attention (MQA)

    Motivation: KV cache size is proportional to num_kv_heads * d_k * T.
    Reducing num_kv_heads dramatically reduces KV cache memory at inference time
    with relatively small quality loss.

    Args:
        d_model:      Total model dimension.
        num_heads:    Number of query heads.
        num_kv_heads: Number of key/value heads. Must divide num_heads evenly.
        dropout:      Dropout on attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # Query heads per KV head
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # Q: full num_heads heads
        self.q_proj = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        # K, V: only num_kv_heads heads
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        Self-attention variant. x: (B, T, d_model)
        Returns: (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        # Q: (B, num_heads, T, d_k)
        # K: (B, num_kv_heads, T, d_k)
        # V: (B, num_kv_heads, T, d_k)

        # Expand K and V to match query heads by repeating each KV head num_groups times.
        # (B, num_kv_heads, T, d_k) -> (B, num_heads, T, d_k)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        # Standard attention
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, num_heads, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_against_pytorch_mha():
    """
    Test our MHA against torch.nn.MultiheadAttention.
    We copy weights to ensure identical initialisation.
    """
    print("Test 1: Compare against nn.MultiheadAttention (identical weights)")
    B, T, d_model, num_heads = 2, 10, 64, 4

    # Our implementation
    our_mha = MultiHeadAttention(d_model, num_heads, bias=True)

    # PyTorch reference
    torch_mha = nn.MultiheadAttention(d_model, num_heads, bias=True, batch_first=True)

    # PyTorch stores Q/K/V weights as a single concatenated matrix: in_proj_weight
    # Shape: (3*d_model, d_model) = [W_Q; W_K; W_V]
    # We need to copy our weights into this format.
    with torch.no_grad():
        in_proj_weight = torch.cat([
            our_mha.q_proj.weight,
            our_mha.k_proj.weight,
            our_mha.v_proj.weight,
        ], dim=0)
        in_proj_bias = torch.cat([
            our_mha.q_proj.bias,
            our_mha.k_proj.bias,
            our_mha.v_proj.bias,
        ])
        torch_mha.in_proj_weight.copy_(in_proj_weight)
        torch_mha.in_proj_bias.copy_(in_proj_bias)
        torch_mha.out_proj.weight.copy_(our_mha.out_proj.weight)
        torch_mha.out_proj.bias.copy_(our_mha.out_proj.bias)

    X = torch.randn(B, T, d_model)

    out_ours, _ = our_mha(X, X, X)  # self-attention
    out_torch, _ = torch_mha(X, X, X)

    max_diff = (out_ours - out_torch).abs().max().item()
    print(f"  Max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff}"
    print("  PASSED")


def test_causal_mha():
    """Test that causal MHA prevents attending to future tokens."""
    print("Test 2: Causal masking in MHA")
    B, T, d_model, num_heads = 2, 8, 32, 4

    mha = MultiHeadAttention(d_model, num_heads)
    X = torch.randn(B, T, d_model)

    out, weights = mha(X, X, X, is_causal=True, need_weights=True)

    assert weights is not None
    for i in range(T):
        for j in range(i + 1, T):
            w = weights[:, :, i, j].abs().max().item()
            assert w < 1e-6, f"Non-zero attention from pos {i} to future {j}: {w}"
    print(f"  Output shape: {out.shape}")
    print("  All future attention weights are zero")
    print("  PASSED")


def test_padding_mask_mha():
    """Test that padded positions receive zero attention weight."""
    print("Test 3: Padding mask in MHA")
    B, T, d_model, num_heads = 3, 10, 32, 4

    mha = MultiHeadAttention(d_model, num_heads)
    X = torch.randn(B, T, d_model)

    # Lengths: [10, 7, 5] — last 0/3/5 positions are padding
    lengths = torch.tensor([10, 7, 5])
    # key_padding_mask: (B, T), True = padding
    positions = torch.arange(T).unsqueeze(0)     # (1, T)
    pad_mask = positions >= lengths.unsqueeze(1) # (B, T)

    _, weights = mha(X, X, X, key_padding_mask=pad_mask, need_weights=True)

    for b, length in enumerate(lengths.tolist()):
        w = weights[b, :, :, length:].abs().max().item()
        assert w < 1e-6, \
            f"Non-zero attention to padding in batch {b}: {w}"
    print(f"  Sequence lengths: {lengths.tolist()}")
    print("  Padding positions have zero attention weight in all batches")
    print("  PASSED")


def test_output_shape():
    """Test that output shape is correct for various configs."""
    print("Test 4: Output shapes")
    configs = [
        (2, 10, 64, 4),
        (1, 1, 128, 8),
        (4, 20, 256, 16),
    ]
    for B, T, d_model, H in configs:
        mha = MultiHeadAttention(d_model, H)
        X = torch.randn(B, T, d_model)
        out, _ = mha(X, X, X)
        assert out.shape == (B, T, d_model), \
            f"Wrong shape for config ({B},{T},{d_model},{H}): {out.shape}"
        print(f"  Config (B={B}, T={T}, d={d_model}, H={H}): output {out.shape}  OK")
    print("  PASSED")


def test_gqa():
    """Test Grouped Query Attention with different group sizes."""
    print("Test 5: Grouped Query Attention (GQA)")
    B, T, d_model = 2, 12, 64

    # MQA: 1 KV head
    gqa_mqa = GroupedQueryAttention(d_model, num_heads=8, num_kv_heads=1)
    X = torch.randn(B, T, d_model)
    out_mqa = gqa_mqa(X)
    assert out_mqa.shape == (B, T, d_model)
    print(f"  MQA (8 Q heads, 1 KV head): output shape {out_mqa.shape}  OK")

    # GQA: 2 KV heads
    gqa_2 = GroupedQueryAttention(d_model, num_heads=8, num_kv_heads=2)
    out_gqa = gqa_2(X)
    assert out_gqa.shape == (B, T, d_model)
    print(f"  GQA (8 Q heads, 2 KV heads): output shape {out_gqa.shape}  OK")

    # MHA equivalent: num_kv_heads == num_heads
    gqa_mha = GroupedQueryAttention(d_model, num_heads=8, num_kv_heads=8)
    out_mha = gqa_mha(X)
    assert out_mha.shape == (B, T, d_model)
    print(f"  GQA as MHA (8 Q heads, 8 KV heads): output shape {out_mha.shape}  OK")

    print("  PASSED")


def test_gradient_flow_mha():
    """Verify gradients flow to all projection weights."""
    print("Test 6: Gradient flow through MHA")
    B, T, d_model, H = 2, 6, 32, 4
    mha = MultiHeadAttention(d_model, H)
    X = torch.randn(B, T, d_model)

    out, _ = mha(X, X, X)
    out.sum().backward()

    for name, param in mha.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().max() > 0, f"Zero gradient for {name}"
        print(f"  {name}: grad norm {param.grad.norm():.4f}")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge 02: Multi-Head Attention")
    print("=" * 60)
    print()

    test_against_pytorch_mha()
    print()
    test_causal_mha()
    print()
    test_padding_mask_mha()
    print()
    test_output_shape()
    print()
    test_gqa()
    print()
    test_gradient_flow_mha()
    print()
    print("All tests passed.")
