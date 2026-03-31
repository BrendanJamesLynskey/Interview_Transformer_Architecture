"""
Challenge 06: KV Cache for Autoregressive Generation
=====================================================

Topic: Transformer inference optimisation
Difficulty: Intermediate / Advanced

Problem
-------
During autoregressive text generation a decoder-only Transformer produces one
token per forward pass.  The naive approach recomputes Keys and Values for every
previously seen token at each new step, giving O(T^2) total work for a sequence
of length T.  The KV cache stores those tensors after they are first computed so
each new step only needs to compute K/V for the single new token and then
concatenates the result with the cache.

Your tasks
----------
1. Implement a minimal decoder-only Transformer (2 layers, causal self-attention).
2. Implement `generate_no_cache` -- standard greedy decode that recomputes
   attention from scratch at every step.
3. Implement `generate_with_cache` -- greedy decode that maintains a KV cache,
   only computing new K/V projections for the incoming token and concatenating
   with cached tensors.
4. Benchmark both approaches and print the wall-clock speedup.

Key insight
-----------
Without cache: each step i processes a (1 x i) sequence -> total FLOPs ~ O(T^2 * d).
With cache:    each step i projects ONE token and does ONE row of attention -> O(T * d).

The memory trade-off: the cache holds 2 * n_layers * n_heads * T * head_dim floats.
For a 7B-parameter model with 32 layers, 32 heads, head_dim=128, and T=2048 that is
roughly 2 * 32 * 32 * 128 * 2048 * 2 bytes (fp16) ≈ 16 GB -- non-trivial!

Dependencies: torch only (no transformers library required).
"""

import time
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional KV cache support.

    The `past_kv` argument enables the cached path:
      - If None:   compute K, V for the full input sequence (no-cache path).
      - If tuple:  (cached_k, cached_v) already computed for previous tokens;
                   compute K, V only for the new token(s), then concatenate.

    Returns (output, (new_k, new_v)) so callers can update the cache.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Single projection matrices for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Fixed causal mask (upper-triangular -inf) -- only needed for the
        # no-cache path where we attend over a full sequence at once.
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1
            ),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, n_heads, T, head_dim)"""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, n_heads, T, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, T, head_dim) -> (B, T, d_model)"""
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x       : (B, T_new, d_model)  -- new token(s) only when using cache
        past_kv : None | (k_cache, v_cache) where each is (B, n_heads, T_past, head_dim)

        Returns
        -------
        output  : (B, T_new, d_model)
        new_kv  : (k_full, v_full) -- updated cache tensors
        """
        B, T_new, _ = x.shape

        # Project new tokens to Q, K, V
        q = self._split_heads(self.W_q(x))  # (B, n_heads, T_new, head_dim)
        k_new = self._split_heads(self.W_k(x))
        v_new = self._split_heads(self.W_v(x))

        # ------------------------------------------------------------------
        # KV cache concatenation
        # ------------------------------------------------------------------
        if past_kv is not None:
            k_cache, v_cache = past_kv
            # Append new K/V to cached K/V along the sequence dimension
            k_full = torch.cat([k_cache, k_new], dim=2)  # (B, n_heads, T_past+T_new, head_dim)
            v_full = torch.cat([v_cache, v_new], dim=2)
        else:
            k_full = k_new
            v_full = v_new

        T_full = k_full.size(2)  # total key/value length

        # ------------------------------------------------------------------
        # Scaled dot-product attention
        # attn_scores: (B, n_heads, T_new, T_full)
        # ------------------------------------------------------------------
        attn_scores = torch.matmul(q, k_full.transpose(-2, -1)) / self.scale

        # Apply causal mask.  When using cache, each query position can
        # attend to all cached positions -- no masking needed for T_new == 1.
        # For the full-sequence (no-cache) case we apply the standard mask.
        if past_kv is None and T_new > 1:
            # Slice the precomputed mask to the actual sequence length
            mask = self.causal_mask[:T_new, :T_full]
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_full)          # (B, n_heads, T_new, head_dim)

        out = self._merge_heads(out)                       # (B, T_new, d_model)
        out = self.W_o(out)

        return out, (k_full, v_full)


class FeedForward(nn.Module):
    """Standard two-layer FFN with GELU activation (as used in GPT-2 / GPT-3)."""

    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion, bias=True),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm decoder block: LayerNorm -> Attention -> residual,
    then LayerNorm -> FFN -> residual.

    Pre-norm (norm BEFORE the sublayer) is used by most modern LLMs (GPT-2+,
    LLaMA, etc.) because it provides more stable gradients than post-norm.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Attention sublayer (with residual)
        normed = self.ln1(x)
        attn_out, new_kv = self.attn(normed, past_kv=past_kv)
        x = x + attn_out

        # FFN sublayer (with residual)
        x = x + self.ffn(self.ln2(x))

        return x, new_kv


class DecoderOnlyTransformer(nn.Module):
    """
    Minimal decoder-only Transformer suitable for autoregressive generation.

    Architecture
    ------------
    - Token embedding table (vocab_size -> d_model)
    - Learned positional embeddings (max_seq_len -> d_model)
    - N stacked TransformerBlock layers (pre-norm)
    - Final LayerNorm
    - Linear head (d_model -> vocab_size), weight-tied to embedding

    This mirrors the GPT-2 architecture at a small scale.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_seq_len) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)

        # Weight tying: share token embedding and output projection weights.
        # This reduces parameters and empirically improves perplexity.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Parameters
        ----------
        token_ids : (B, T)  -- token indices for the NEW tokens only when using cache
        past_kvs  : list of (k_cache, v_cache) per layer, or None

        Returns
        -------
        logits   : (B, T, vocab_size)
        new_kvs  : updated list of (k, v) per layer
        """
        B, T = token_ids.shape

        # Determine positional offset when cache is active
        if past_kvs is not None:
            # The cache already holds T_past steps; new tokens start at T_past
            offset = past_kvs[0][0].size(2)  # T_past from layer 0's K cache
        else:
            offset = 0

        positions = torch.arange(offset, offset + T, device=token_ids.device)

        x = self.token_embed(token_ids) + self.pos_embed(positions)

        new_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, past_kv=layer_past)
            new_kvs.append(new_kv)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits, new_kvs


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_no_cache(
    model: DecoderOnlyTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Greedy autoregressive generation WITHOUT KV cache.

    At every step we feed the ENTIRE sequence generated so far back into
    the model.  This recomputes K and V for every previous token at each step,
    resulting in O(T^2) total attention work.

    Parameters
    ----------
    prompt         : (1, T_prompt) initial token ids
    max_new_tokens : number of tokens to generate

    Returns
    -------
    (1, T_prompt + max_new_tokens) completed token sequence
    """
    model.eval()
    with torch.no_grad():
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Full sequence forward pass -- no cache argument
            logits, _ = model(generated, past_kvs=None)

            # Greedy: take the argmax of the last position's logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def generate_with_cache(
    model: DecoderOnlyTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Greedy autoregressive generation WITH KV cache.

    Step 0 (prefill): feed the entire prompt to build an initial KV cache.
    Steps 1..T (decode): feed only the single new token; concatenate new K/V
    with the cache inside each attention layer.

    This reduces the per-step attention work from O(T) to O(1) new projections
    plus one row of dot-products against the cached K/V.

    Parameters
    ----------
    prompt         : (1, T_prompt) initial token ids
    max_new_tokens : number of tokens to generate

    Returns
    -------
    (1, T_prompt + max_new_tokens) completed token sequence
    """
    model.eval()
    with torch.no_grad():
        # ----- Prefill phase -----
        # Process the full prompt once to populate the KV cache.
        logits, kv_cache = model(prompt, past_kvs=None)

        # Greedily pick the first new token from the last prompt position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        generated = torch.cat([prompt, next_token], dim=1)

        # ----- Decode phase -----
        for _ in range(max_new_tokens - 1):
            # Feed ONLY the most recent token; the cache covers everything before it
            logits, kv_cache = model(next_token, past_kvs=kv_cache)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def verify_outputs_match(
    model: DecoderOnlyTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
) -> bool:
    """
    Both generation paths must produce identical token sequences because
    they implement the same mathematical operation -- the cache is purely
    a computational shortcut, not an approximation.
    """
    out_no_cache = generate_no_cache(model, prompt, max_new_tokens)
    out_with_cache = generate_with_cache(model, prompt, max_new_tokens)
    match = torch.equal(out_no_cache, out_with_cache)
    if match:
        print("Correctness check PASSED: both methods produce identical outputs.")
    else:
        print("Correctness check FAILED: outputs differ!")
        print("  No cache:", out_no_cache[0].tolist())
        print("  With cache:", out_with_cache[0].tolist())
    return match


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    model: DecoderOnlyTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    n_runs: int = 5,
) -> dict[str, float]:
    """
    Time both generation strategies over `n_runs` runs and return mean
    wall-clock times in milliseconds.

    We do one warm-up run before timing to avoid including Python JIT / CUDA
    initialisation costs in the measurements.
    """
    device = prompt.device

    def timed_run(fn, *args) -> float:
        # Warm-up
        fn(*args)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            fn(*args)
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms
        return elapsed

    t_no_cache = timed_run(generate_no_cache, model, prompt, max_new_tokens)
    t_with_cache = timed_run(generate_with_cache, model, prompt, max_new_tokens)

    return {"no_cache_ms": t_no_cache, "with_cache_ms": t_with_cache}


# ---------------------------------------------------------------------------
# Memory analysis helper
# ---------------------------------------------------------------------------

def kv_cache_memory_bytes(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 4,  # fp32 by default; use 2 for fp16
) -> int:
    """
    Compute the KV cache memory footprint in bytes.

    Formula: 2 (K and V) * n_layers * batch_size * n_heads * seq_len * head_dim * dtype_bytes
    """
    return 2 * n_layers * batch_size * n_heads * seq_len * head_dim * dtype_bytes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # -----------------------------------------------------------------------
    # Model configuration (small but non-trivial)
    # -----------------------------------------------------------------------
    VOCAB_SIZE = 256
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 3
    MAX_SEQ_LEN = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # -----------------------------------------------------------------------
    # Prompt and generation settings
    # -----------------------------------------------------------------------
    PROMPT_LEN = 64
    MAX_NEW_TOKENS = 128

    prompt = torch.randint(0, VOCAB_SIZE, (1, PROMPT_LEN), device=device)

    print(f"Prompt length    : {PROMPT_LEN} tokens")
    print(f"New tokens       : {MAX_NEW_TOKENS} tokens")
    print(f"Total sequence   : {PROMPT_LEN + MAX_NEW_TOKENS} tokens\n")

    # -----------------------------------------------------------------------
    # 1. Correctness verification
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)
    verify_outputs_match(model, prompt, MAX_NEW_TOKENS)

    # -----------------------------------------------------------------------
    # 2. Benchmark
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BENCHMARK (5 runs each)")
    print("=" * 60)

    results = benchmark(model, prompt, MAX_NEW_TOKENS, n_runs=5)

    t_no = results["no_cache_ms"]
    t_kv = results["with_cache_ms"]
    speedup = t_no / t_kv

    print(f"No cache  : {t_no:8.2f} ms")
    print(f"KV cache  : {t_kv:8.2f} ms")
    print(f"Speedup   : {speedup:.2f}x")

    # -----------------------------------------------------------------------
    # 3. Memory analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KV CACHE MEMORY ANALYSIS")
    print("=" * 60)

    for dtype_label, dtype_bytes in [("fp32", 4), ("fp16", 2), ("int8", 1)]:
        mem = kv_cache_memory_bytes(
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            head_dim=D_MODEL // N_HEADS,
            seq_len=PROMPT_LEN + MAX_NEW_TOKENS,
            batch_size=1,
            dtype_bytes=dtype_bytes,
        )
        print(f"  {dtype_label:5s}: {mem / 1024:.1f} KB  ({mem:,} bytes)")

    # Scale-up example: LLaMA-2 7B dimensions
    print("\n  Scale-up example (LLaMA-2 7B, T=4096, batch=1, fp16):")
    llama_mem = kv_cache_memory_bytes(
        n_layers=32, n_heads=32, head_dim=128, seq_len=4096, batch_size=1, dtype_bytes=2
    )
    print(f"    KV cache: {llama_mem / 1024**3:.2f} GB")

    # -----------------------------------------------------------------------
    # 4. Demonstrate scaling behaviour
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SCALING: generation time vs. sequence length")
    print("=" * 60)
    print(f"{'Seq len':>10}  {'No cache (ms)':>15}  {'KV cache (ms)':>15}  {'Speedup':>10}")
    print("-" * 58)

    for total_len in [32, 64, 128, 256]:
        if total_len <= PROMPT_LEN:
            continue
        new_toks = total_len - PROMPT_LEN
        r = benchmark(model, prompt, new_toks, n_runs=3)
        spd = r["no_cache_ms"] / r["with_cache_ms"]
        print(
            f"{total_len:>10}  {r['no_cache_ms']:>15.2f}  {r['with_cache_ms']:>15.2f}  {spd:>9.2f}x"
        )

    print("\nDone.")
    print(
        "\nKey takeaway: KV cache speedup grows with sequence length because the"
        "\nno-cache path recomputes O(T^2) attention work, while the cache path"
        "\ndoes O(T) total work -- one new K/V projection per step."
    )
