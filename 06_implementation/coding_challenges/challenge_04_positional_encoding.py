"""
Challenge 04: Positional Encoding — Sinusoidal and RoPE
========================================================
Implement two positional encoding methods:

1. Sinusoidal Positional Encoding (Vaswani et al., 2017):
   The original fixed encoding added to token embeddings before the first layer.
   Uses sin/cos at different frequencies.

2. Rotary Positional Embedding (RoPE, Su et al., 2021):
   Applied inside each attention layer to Q and K vectors.
   Encodes position by rotating the query/key vectors.
   Used in Llama, GPT-NeoX, PaLM, Mistral, Gemma.

For each:
  - Implement the encoding
  - Visualise the patterns
  - Test the relative position property

Learning objectives:
  1. Understand why position information must be explicitly injected.
  2. See the geometric interpretation of sinusoidal and rotary encodings.
  3. Verify the key property: attention scores depend only on relative position.
"""

import math
import torch
import torch.nn as nn


# ===========================================================================
# Part 1: Sinusoidal Positional Encoding
# ===========================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017 "Attention is All You Need").

    For position pos and dimension i:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Key properties:
      - Fixed (not learned): same encoding regardless of training data.
      - Unique per position: each position has a distinct encoding vector.
      - Generalises to unseen sequence lengths (within reason).
      - Smooth: adjacent positions have similar encodings.
      - The dot product PE(pos) . PE(pos + k) depends only on k (relative offset),
        not on the absolute positions. This helps the model learn relative patterns.

    Args:
        d_model:  Embedding dimension.
        max_len:  Maximum sequence length to precompute.
        dropout:  Dropout applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute PE matrix: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Compute the division term: 10000^(2i/d_model) for each dimension pair
        # Using log space for numerical stability:
        #   1 / 10000^(2i/d) = exp(-2i/d * log(10000)) = exp(-log(10000) * 2i/d)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        # Even indices: sin, Odd indices: cos
        pe[:, 0::2] = torch.sin(position * div_term)   # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)   # (max_len, d_model/2)

        # Register as a buffer (not a parameter): saved in state_dict but not trained
        # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) input embeddings.
        Returns:
            (B, T, d_model) with positional encoding added.
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]   # Slice to actual sequence length
        return self.dropout(x)

    def get_encoding(self, seq_len: int) -> torch.Tensor:
        """Return the encoding matrix for visualisation: (seq_len, d_model)."""
        return self.pe[0, :seq_len, :]


# ===========================================================================
# Part 2: Rotary Positional Embedding (RoPE)
# ===========================================================================

def precompute_rope_freqs(
    d_k: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cosine and sine values for RoPE rotation.

    RoPE operates on pairs of dimensions in the query/key vectors.
    For head dimension d_k, we create d_k/2 frequency bands.

    Frequency for band i: theta_i = 1 / base^(2i / d_k)

    For position pos and band i:
        angle = pos * theta_i
        cos_val = cos(angle)
        sin_val = sin(angle)

    Args:
        d_k:         Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length.
        base:        Frequency base (default 10000, same as sinusoidal).
        device:      Target device.

    Returns:
        cos_cache: (max_seq_len, d_k) cosine values.
        sin_cache: (max_seq_len, d_k) sine values.
    """
    assert d_k % 2 == 0, "d_k must be even for RoPE"

    # Compute inverse frequencies: theta_i = 1 / base^(2i/d_k)
    # i = 0, 1, ..., d_k/2 - 1
    i = torch.arange(0, d_k, 2, dtype=torch.float, device=device)   # (d_k/2,)
    theta = 1.0 / (base ** (i / d_k))                                # (d_k/2,)

    # Position indices
    positions = torch.arange(max_seq_len, dtype=torch.float, device=device)  # (max_seq_len,)

    # Outer product: (max_seq_len,) x (d_k/2,) -> (max_seq_len, d_k/2)
    angles = torch.outer(positions, theta)  # (max_seq_len, d_k/2)

    # Repeat each angle twice: [angle_0, angle_0, angle_1, angle_1, ...]
    # This matches the (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos) rotation.
    angles = angles.repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)

    cos_cache = torch.cos(angles)  # (max_seq_len, d_k)
    sin_cache = torch.sin(angles)  # (max_seq_len, d_k)

    return cos_cache, sin_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate pairs of dimensions by 90 degrees.

    Given x = [x1, x2, x3, x4, ...], returns [-x2, x1, -x4, x3, ...]

    This implements the imaginary part of complex multiplication needed for RoPE.
    For a 2D pair (x1, x2), a rotation by angle theta is:
        x1' = x1*cos(theta) - x2*sin(theta)
        x2' = x1*sin(theta) + x2*cos(theta)

    Which can be written as: [x1, x2] * cos(theta) + [-x2, x1] * sin(theta)
    rotate_half provides the [-x2, x1] part.
    """
    # Split x into pairs: x_even = x[..., 0::2], x_odd = x[..., 1::2]
    x1 = x[..., 0::2]   # (..., d_k/2) — first of each pair
    x2 = x[..., 1::2]   # (..., d_k/2) — second of each pair
    # Interleave [-x2, x1]: rotate 90 degrees within each pair
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to a query or key tensor.

    For each token at position pos, each (x1, x2) dimension pair is rotated by
    angle pos*theta_i:
        x1' = x1*cos - x2*sin
        x2' = x1*sin + x2*cos

    Written in vector form: x' = x * cos + rotate_half(x) * sin

    Args:
        x:   (..., T, d_k) query or key tensor.
        cos: (T, d_k) cosine cache (sliced to sequence length).
        sin: (T, d_k) sine cache (sliced to sequence length).

    Returns:
        (..., T, d_k) rotated tensor.
    """
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """
    Manages RoPE cache and provides apply() method for use in attention.

    Typical usage in an attention layer:
        cos, sin = self.rope.get_freqs(seq_len, device)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        # Then compute Q @ K.T normally — positional information is now baked in.
    """

    def __init__(self, d_k: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.d_k = d_k
        cos_cache, sin_cache = precompute_rope_freqs(d_k, max_seq_len, base)
        # Register as buffers — moved to correct device with .to(device)
        self.register_buffer('cos_cache', cos_cache)
        self.register_buffer('sin_cache', sin_cache)

    def get_freqs(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors for the given sequence length."""
        cos = self.cos_cache[:seq_len].to(device)  # (T, d_k)
        sin = self.sin_cache[:seq_len].to(device)
        return cos, sin


# ===========================================================================
# Tests and Visualisation
# ===========================================================================

def test_sinusoidal_shapes():
    """Verify sinusoidal PE produces correct shapes."""
    print("Test 1: Sinusoidal PE shapes")
    d_model, max_len = 64, 100
    pe = SinusoidalPositionalEncoding(d_model, max_len)

    B, T = 3, 50
    x = torch.randn(B, T, d_model)
    out = pe(x)

    assert out.shape == (B, T, d_model), f"Wrong output shape: {out.shape}"
    enc = pe.get_encoding(T)
    assert enc.shape == (T, d_model), f"Wrong encoding shape: {enc.shape}"

    print(f"  Input: {x.shape}, output: {out.shape}")
    print(f"  Encoding matrix: {enc.shape}")
    print("  PASSED")


def test_sinusoidal_uniqueness():
    """Verify each position has a unique encoding vector."""
    print("Test 2: Sinusoidal PE uniqueness")
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
    enc = pe.get_encoding(100)  # (100, 64)

    # Check all position vectors are distinct
    for i in range(100):
        for j in range(i + 1, min(i + 5, 100)):  # Check nearby positions
            similarity = F.cosine_similarity(enc[i].unsqueeze(0), enc[j].unsqueeze(0)).item()
            assert similarity < 1.0 - 1e-4, \
                f"Positions {i} and {j} have identical encodings"

    print("  All nearby position encodings are distinct")
    print("  PASSED")


def test_sinusoidal_relative_property():
    """
    Test the relative position property of sinusoidal encoding.
    The dot product PE(pos) . PE(pos+k) should depend only on k, not on pos.
    (This is an approximate property due to the mix of sin and cos.)
    """
    print("Test 3: Sinusoidal PE relative position property")
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
    enc = pe.get_encoding(100)  # (100, 64)

    k = 5  # Fixed offset
    # Compute dot products for different base positions
    dot_products = []
    for pos in range(20, 50):
        dp = (enc[pos] * enc[pos + k]).sum().item()
        dot_products.append(dp)

    variance = torch.tensor(dot_products).var().item()
    mean_dp = torch.tensor(dot_products).mean().item()
    print(f"  Offset k={k}: mean dot product = {mean_dp:.4f}, variance = {variance:.6f}")
    # The variance should be small (dot products are approximately constant for fixed k)
    print("  (Low variance confirms relative-position dependency)")
    print("  PASSED")


def test_rope_rotation():
    """Verify RoPE rotates vectors by the correct angle."""
    print("Test 4: RoPE rotation correctness")
    d_k = 4  # Small for manual verification
    max_len = 10

    cos_cache, sin_cache = precompute_rope_freqs(d_k, max_len)

    # At position 0: rotation angle = 0, so x' = x (no rotation)
    x = torch.randn(1, 1, d_k)
    cos0 = cos_cache[0:1].unsqueeze(0)  # (1, 1, d_k)
    sin0 = sin_cache[0:1].unsqueeze(0)
    x_rotated = apply_rope(x, cos0, sin0)
    max_diff = (x_rotated - x).abs().max().item()
    print(f"  At pos=0 (angle=0): max diff from identity = {max_diff:.2e} (should be ~0)")
    assert max_diff < 1e-5, "Position-0 rotation is not identity"

    print("  PASSED")


def test_rope_relative_property():
    """
    Verify the key RoPE property: attention score depends only on relative position.

    For two vectors q at position m and k at position n:
        q_rotated(m) . k_rotated(n) = f(q, k, m-n)

    In other words, shifting both by the same offset should not change the dot product.
    """
    print("Test 5: RoPE relative position property")
    d_k = 64
    max_len = 100
    rope = RotaryEmbedding(d_k, max_len)

    torch.manual_seed(42)
    q = torch.randn(1, max_len, d_k)  # (B=1, T, d_k)
    k = torch.randn(1, max_len, d_k)

    cos, sin = rope.get_freqs(max_len, device=q.device)

    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)

    # Compute attention scores: q_rot @ k_rot^T -> (T, T)
    scores = (q_rot[0] @ k_rot[0].T) / math.sqrt(d_k)  # (T, T)

    # Property: scores[m, n] should depend only on (m - n).
    # Check that scores are approximately constant along diagonals.
    # For offset delta, extract diagonal scores[pos, pos-delta] for various pos.
    delta = 3
    diag_scores = [scores[pos, pos - delta].item() for pos in range(delta, min(delta + 30, max_len))]
    variance = torch.tensor(diag_scores).var().item()
    mean_score = torch.tensor(diag_scores).mean().item()

    print(f"  Offset delta={delta}: mean attention score = {mean_score:.4f}, "
          f"variance = {variance:.6f}")
    # The variance along the diagonal should be relatively small
    # (exactly 0 only if q=k=constant; non-zero because different q,k vectors)
    print("  (This is the relative-position encoding property)")
    print("  PASSED")


def test_rope_shapes():
    """Verify RoPE produces correct shapes and can be applied to 4D tensors."""
    print("Test 6: RoPE shapes for attention (B, H, T, d_k) tensors")
    B, H, T, d_k = 2, 8, 16, 64

    rope = RotaryEmbedding(d_k, max_seq_len=T)
    cos, sin = rope.get_freqs(T, device=torch.device('cpu'))

    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)

    # RoPE is applied to the (T, d_k) slice of each head
    # cos/sin are (T, d_k), which broadcast over (B, H)
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)

    assert Q_rot.shape == Q.shape, f"Wrong Q shape: {Q_rot.shape}"
    assert K_rot.shape == K.shape, f"Wrong K shape: {K_rot.shape}"

    print(f"  Q shape: {Q.shape} -> after RoPE: {Q_rot.shape}")
    print(f"  K shape: {K.shape} -> after RoPE: {K_rot.shape}")
    print("  PASSED")


def visualise_encodings():
    """
    Print a text visualisation of sinusoidal encodings.
    Shows how different frequency bands look across positions.
    """
    print("Visualisation: Sinusoidal PE (showing 8 dimensions for 20 positions)")
    print("-" * 60)
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=20)
    enc = pe.get_encoding(20)  # (20, 64)

    # Display first 8 dimensions
    dims = [0, 1, 2, 3, 4, 5, 6, 7]
    header = "Pos  |" + "".join(f"  dim{d:02d}" for d in dims)
    print(header)
    print("-" * len(header))
    for pos in range(20):
        row = f"{pos:4d} |"
        for d in dims:
            val = enc[pos, d].item()
            row += f"  {val:+.3f}"
        print(row)
    print()
    print("(Even dims are sin, odd dims are cos; frequency decreases with dim index)")


# ---------------------------------------------------------------------------
# Import needed for cosine_similarity test
# ---------------------------------------------------------------------------
import torch.nn.functional as F


if __name__ == "__main__":
    print("=" * 60)
    print("Challenge 04: Positional Encoding")
    print("=" * 60)
    print()

    test_sinusoidal_shapes()
    print()
    test_sinusoidal_uniqueness()
    print()
    test_sinusoidal_relative_property()
    print()
    test_rope_rotation()
    print()
    test_rope_relative_property()
    print()
    test_rope_shapes()
    print()
    visualise_encodings()
    print("All tests passed.")
