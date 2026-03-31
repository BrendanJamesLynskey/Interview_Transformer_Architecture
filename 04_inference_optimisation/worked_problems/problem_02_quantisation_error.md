# Worked Problem 02 — Quantisation Error: INT8 Symmetric and INT4 Asymmetric

## Problem Statement

You are given the following $3 \times 4$ weight matrix in FP32:

$$
W = \begin{bmatrix}
 0.32 & -1.47 &  0.08 &  2.13 \\
-0.61 &  0.95 & -2.84 &  0.17 \\
 1.22 & -0.43 &  0.76 & -1.08
\end{bmatrix}
$$

And a FP32 input vector of length 4:

$$
x = \begin{bmatrix} 1.0 \\ -0.5 \\ 2.0 \\ 0.3 \end{bmatrix}
$$

**Tasks:**

1. Quantise $W$ to **INT8 using symmetric (per-tensor) quantisation**. Show the
   scale factor, quantised matrix, and dequantised matrix. Compute the per-element
   quantisation error.

2. Quantise $W$ to **INT4 using asymmetric (per-tensor) quantisation**. Show the
   scale and zero-point, quantised matrix, dequantised matrix, and per-element
   error.

3. Compute the matrix-vector product $y = Wx$ in:
   - FP32 (reference)
   - Dequantised INT8
   - Dequantised INT4
   Show the error in each output element.

4. Explain why per-channel quantisation reduces error for weight matrices.

---

## Background: Quantisation Schemes

### Symmetric INT8

Maps the range $[-\alpha, +\alpha]$ (where $\alpha = \max |W_{ij}|$) to $[-127, 127]$:

$$
s = \frac{\alpha}{127}, \qquad W_q = \text{round}\!\left(\frac{W}{s}\right), \qquad W_\text{deq} = s \cdot W_q
$$

The zero-point is fixed at 0 (symmetry), so no addition is required at inference —
only a scalar multiply to dequantise.

### Asymmetric INT4

Maps the range $[W_\min, W_\max]$ to $[0, 15]$ (unsigned 4-bit integers):

$$
s = \frac{W_\max - W_\min}{15}, \qquad z = \text{round}\!\left(\frac{-W_\min}{s}\right), \qquad W_q = \text{clamp}\!\left(\text{round}\!\left(\frac{W}{s}\right) + z,\; 0,\; 15\right)
$$

Dequantisation:

$$
W_\text{deq} = s \cdot (W_q - z)
$$

---

## Part 1 — INT8 Symmetric Quantisation

### Step 1: Compute scale factor

$$
\alpha = \max |W_{ij}| = \max(|0.32|, |-1.47|, \ldots, |-1.08|) = |-2.84| = 2.84
$$

$$
s_8 = \frac{2.84}{127} = 0.022362\ldots \approx 0.02236
$$

### Step 2: Quantise each element

$$
W_q^{(8)} = \text{round}\!\left(\frac{W}{0.02236}\right)
$$

Computing row by row:

**Row 1:**

| $W$ | $W / s$ | $\text{round}$ |
|-----|---------|----------------|
| 0.32 | $0.32 / 0.02236 = 14.31$ | 14 |
| -1.47 | $-1.47 / 0.02236 = -65.74$ | -66 |
| 0.08 | $0.08 / 0.02236 = 3.58$ | 4 |
| 2.13 | $2.13 / 0.02236 = 95.26$ | 95 |

**Row 2:**

| $W$ | $W / s$ | $\text{round}$ |
|-----|---------|----------------|
| -0.61 | $-0.61 / 0.02236 = -27.28$ | -27 |
| 0.95 | $0.95 / 0.02236 = 42.49$ | 42 |
| -2.84 | $-2.84 / 0.02236 = -127.00$ | -127 |
| 0.17 | $0.17 / 0.02236 = 7.60$ | 8 |

**Row 3:**

| $W$ | $W / s$ | $\text{round}$ |
|-----|---------|----------------|
| 1.22 | $1.22 / 0.02236 = 54.56$ | 55 |
| -0.43 | $-0.43 / 0.02236 = -19.23$ | -19 |
| 0.76 | $0.76 / 0.02236 = 33.99$ | 34 |
| -1.08 | $-1.08 / 0.02236 = -48.30$ | -48 |

$$
W_q^{(8)} = \begin{bmatrix}
 14 & -66 &   4 &  95 \\
-27 &  42 & -127 &   8 \\
 55 & -19 &  34 & -48
\end{bmatrix}
$$

### Step 3: Dequantise

$$
W_\text{deq}^{(8)} = 0.02236 \times W_q^{(8)}
$$

$$
W_\text{deq}^{(8)} = \begin{bmatrix}
 0.3130 & -1.4758 &  0.0894 &  2.1242 \\
-0.6037 &  0.9391 & -2.8397 &  0.1789 \\
 1.2298 & -0.4248 &  0.7602 & -1.0733
\end{bmatrix}
$$

### Step 4: Per-element quantisation error $\Delta W = W_\text{deq} - W$

$$
\Delta W^{(8)} = \begin{bmatrix}
-0.0070 & -0.0058 &  0.0094 & -0.0058 \\
 0.0063 & -0.0109 & \mathbf{0.0003} &  0.0089 \\
 0.0098 &  0.0052 &  0.0002 &  0.0067
\end{bmatrix}
$$

The element $W_{22} = -2.84$ (which equals $-\alpha$) maps exactly to $-127$ and
has near-zero error. The maximum absolute error is approximately $0.011$, which
is roughly $s_8 / 2 = 0.01118$ as expected (rounding error is at most half a
quantisation step).

---

## Part 2 — INT4 Asymmetric Quantisation

### Step 1: Compute scale and zero-point

$$
W_\min = -2.84, \qquad W_\max = 2.13
$$

$$
s_4 = \frac{W_\max - W_\min}{15} = \frac{2.13 - (-2.84)}{15} = \frac{4.97}{15} = 0.33133
$$

$$
z = \text{round}\!\left(\frac{-W_\min}{s_4}\right) = \text{round}\!\left(\frac{2.84}{0.33133}\right) = \text{round}(8.571) = 9
$$

Verification: the zero FP32 value maps to integer $z = 9$, and
$W_\text{deq}(q=9) = 0.33133 \times (9 - 9) = 0$. Correct.

### Step 2: Quantise each element

$$
W_q^{(4)} = \text{clamp}\!\left(\text{round}\!\left(\frac{W}{0.33133}\right) + 9,\; 0,\; 15\right)
$$

**Row 1:**

| $W$ | $W / s$ | $+z$ | round | clamp(0,15) |
|-----|---------|------|-------|-------------|
| 0.32 | 0.966 | 9.966 | 10 | 10 |
| -1.47 | -4.436 | 4.564 | 5 | 5 |
| 0.08 | 0.241 | 9.241 | 9 | 9 |
| 2.13 | 6.428 | 15.428 | 15 | 15 |

**Row 2:**

| $W$ | $W / s$ | $+z$ | round | clamp(0,15) |
|-----|---------|------|-------|-------------|
| -0.61 | -1.841 | 7.159 | 7 | 7 |
| 0.95 | 2.867 | 11.867 | 12 | 12 |
| -2.84 | -8.571 | 0.429 | 0 | 0 |
| 0.17 | 0.513 | 9.513 | 10 | 10 |

**Row 3:**

| $W$ | $W / s$ | $+z$ | round | clamp(0,15) |
|-----|---------|------|-------|-------------|
| 1.22 | 3.681 | 12.681 | 13 | 13 |
| -0.43 | -1.298 | 7.702 | 8 | 8 |
| 0.76 | 2.294 | 11.294 | 11 | 11 |
| -1.08 | -3.260 | 5.740 | 6 | 6 |

$$
W_q^{(4)} = \begin{bmatrix}
10 &  5 &  9 & 15 \\
 7 & 12 &  0 & 10 \\
13 &  8 & 11 &  6
\end{bmatrix}
$$

### Step 3: Dequantise

$$
W_\text{deq}^{(4)} = 0.33133 \times (W_q^{(4)} - 9)
$$

| Element | $q - z$ | Dequantised |
|---------|---------|-------------|
| $10 - 9 = 1$ | $\times 0.33133$ | 0.3313 |
| $5 - 9 = -4$ | $\times 0.33133$ | -1.3253 |
| $9 - 9 = 0$ | $\times 0.33133$ | 0.0000 |
| $15 - 9 = 6$ | $\times 0.33133$ | 1.9880 |
| $7 - 9 = -2$ | $\times 0.33133$ | -0.6627 |
| $12 - 9 = 3$ | $\times 0.33133$ | 0.9940 |
| $0 - 9 = -9$ | $\times 0.33133$ | -2.9820 |
| $10 - 9 = 1$ | $\times 0.33133$ | 0.3313 |
| $13 - 9 = 4$ | $\times 0.33133$ | 1.3253 |
| $8 - 9 = -1$ | $\times 0.33133$ | -0.3313 |
| $11 - 9 = 2$ | $\times 0.33133$ | 0.6627 |
| $6 - 9 = -3$ | $\times 0.33133$ | -0.9940 |

$$
W_\text{deq}^{(4)} = \begin{bmatrix}
 0.3313 & -1.3253 &  0.0000 &  1.9880 \\
-0.6627 &  0.9940 & -2.9820 &  0.3313 \\
 1.3253 & -0.3313 &  0.6627 & -0.9940
\end{bmatrix}
$$

### Step 4: Per-element error $\Delta W^{(4)} = W_\text{deq}^{(4)} - W$

$$
\Delta W^{(4)} = \begin{bmatrix}
 0.0113 &  0.1447 & -0.0800 & -0.1420 \\
-0.0527 &  0.0440 & -0.1420 &  0.1613 \\
 0.1053 &  0.0987 & -0.0973 &  0.0860
\end{bmatrix}
$$

**Maximum absolute error:** $|0.1613| = 0.161$ — approximately $s_4 / 2 = 0.166$.

The INT4 errors are roughly $0.16 / 0.011 \approx 15\times$ larger than INT8 errors,
consistent with the ratio of quantisation step sizes: $s_4 / s_8 = 0.331 / 0.022 = 15$.

---

## Part 3 — Matrix-Vector Product Comparison

### FP32 Reference: $y = Wx$

$$
x = [1.0,\; -0.5,\; 2.0,\; 0.3]^\top
$$

$$
y_1 = 0.32(1.0) + (-1.47)(-0.5) + 0.08(2.0) + 2.13(0.3)
= 0.32 + 0.735 + 0.16 + 0.639 = \mathbf{1.854}
$$

$$
y_2 = (-0.61)(1.0) + 0.95(-0.5) + (-2.84)(2.0) + 0.17(0.3)
= -0.61 - 0.475 - 5.68 + 0.051 = \mathbf{-6.714}
$$

$$
y_3 = 1.22(1.0) + (-0.43)(-0.5) + 0.76(2.0) + (-1.08)(0.3)
= 1.22 + 0.215 + 1.52 - 0.324 = \mathbf{2.631}
$$

### INT8 Dequantised: $y^{(8)} = W_\text{deq}^{(8)} x$

$$
y_1^{(8)} = 0.3130(1.0) + (-1.4758)(-0.5) + 0.0894(2.0) + 2.1242(0.3)
= 0.3130 + 0.7379 + 0.1788 + 0.6373 = \mathbf{1.867}
$$

$$
y_2^{(8)} = (-0.6037)(1.0) + 0.9391(-0.5) + (-2.8397)(2.0) + 0.1789(0.3)
= -0.6037 - 0.4696 - 5.6794 + 0.0537 = \mathbf{-6.699}
$$

$$
y_3^{(8)} = 1.2298(1.0) + (-0.4248)(-0.5) + 0.7602(2.0) + (-1.0733)(0.3)
= 1.2298 + 0.2124 + 1.5204 - 0.3220 = \mathbf{2.640}
$$

### INT4 Dequantised: $y^{(4)} = W_\text{deq}^{(4)} x$

$$
y_1^{(4)} = 0.3313(1.0) + (-1.3253)(-0.5) + 0.0000(2.0) + 1.9880(0.3)
= 0.3313 + 0.6627 + 0.0000 + 0.5964 = \mathbf{1.590}
$$

$$
y_2^{(4)} = (-0.6627)(1.0) + 0.9940(-0.5) + (-2.9820)(2.0) + 0.3313(0.3)
= -0.6627 - 0.4970 - 5.9640 + 0.0994 = \mathbf{-7.124}
$$

$$
y_3^{(4)} = 1.3253(1.0) + (-0.3313)(-0.5) + 0.6627(2.0) + (-0.9940)(0.3)
= 1.3253 + 0.1657 + 1.3254 - 0.2982 = \mathbf{2.518}
$$

### Error Summary

| Output | FP32 (ref) | INT8 deq | INT8 error | INT4 deq | INT4 error | INT4 rel. error |
|--------|-----------|----------|------------|----------|------------|-----------------|
| $y_1$ | 1.854 | 1.867 | +0.013 | 1.590 | -0.264 | 14.2% |
| $y_2$ | -6.714 | -6.699 | +0.015 | -7.124 | -0.410 | 6.1% |
| $y_3$ | 2.631 | 2.640 | +0.009 | 2.518 | -0.113 | 4.3% |

**Observations:**

1. INT8 errors are small (<1% relative), consistent with the per-element
   quantisation step of $\sim 0.022$.
2. INT4 errors are significantly larger (up to 14% relative), with the error
   in $y_1$ being especially large because the $W_{1,2} = -1.47 \to -1.325$
   error ($\Delta = +0.145$) is multiplied by $x_2 = -0.5$, and
   $W_{1,4} = 2.13 \to 1.988$ error ($\Delta = -0.142$) is multiplied by
   $x_4 = 0.3$. These compound in the same direction for $y_1$.

---

## Part 4 — Why Per-Channel Quantisation Reduces Error

**Per-tensor quantisation** uses a single scale factor $s = \alpha / 127$ where
$\alpha = \max |W_{ij}|$ over the entire matrix. If one element is a large outlier,
$s$ is set by that outlier and all other elements are poorly utilised (spread over
only a fraction of the $[-127, 127]$ range).

**In our matrix**, the element $-2.84$ forces $s = 0.02236$. The element $0.08$
maps to $\text{round}(0.08 / 0.02236) = 4$. If $-2.84$ were absent, we could use
$s = 2.13/127 = 0.01677$, and $0.08$ would map to $\text{round}(0.08/0.01677) = 5$ —
slightly better but similar. However, consider a pathological case where one outlier
is $100\times$ larger than the rest:

**Example.** If $W_\max = 100$ but 99% of elements are in $[-1, 1]$:
- Per-tensor: $s = 100/127 \approx 0.787$. An element of value $0.5$ maps to
  $\text{round}(0.5/0.787) = 1$, representing $0.787$ — an error of $0.287$,
  which is **57%** of the true value.
- Per-channel (the outlier is in its own row): that row uses $s_\text{row} = 100/127$,
  but the other rows use $s = 1.0/127 \approx 0.00787$. The element $0.5$ now maps to
  $\text{round}(0.5/0.00787) = 64$, representing $0.504$ — an error of $0.004$
  (0.8%).

**Per-channel quantisation** assigns a separate scale per output channel (row of
$W$). This ensures each row's dynamic range is independently optimal, virtually
eliminating the outlier problem within a row. The cost is a small amount of
extra metadata (one scale per channel) and a slightly more complex dequantisation
(requires a channel-wise multiply rather than a scalar multiply).

**GPTQ and AWQ** go further: they perform channel-wise INT4 quantisation with
learned scale adjustments that minimise the downstream activation error
$\|Wx - W_\text{deq}x\|$ rather than $\|W - W_\text{deq}\|$ directly.

---

## Code Reference

```python
import numpy as np

W = np.array([
    [ 0.32, -1.47,  0.08,  2.13],
    [-0.61,  0.95, -2.84,  0.17],
    [ 1.22, -0.43,  0.76, -1.08],
], dtype=np.float32)

x = np.array([1.0, -0.5, 2.0, 0.3], dtype=np.float32)

# --- INT8 Symmetric ---
alpha = np.max(np.abs(W))
s8 = alpha / 127.0
Wq8 = np.round(W / s8).astype(np.int8)
Wdeq8 = s8 * Wq8.astype(np.float32)
print("INT8 scale:", s8)
print("INT8 quantised:\n", Wq8)
print("INT8 error (max abs):", np.max(np.abs(Wdeq8 - W)))

# --- INT4 Asymmetric ---
wmin, wmax = W.min(), W.max()
s4 = (wmax - wmin) / 15.0
z4 = round(-wmin / s4)
Wq4 = np.clip(np.round(W / s4 + z4), 0, 15).astype(np.uint8)
Wdeq4 = s4 * (Wq4.astype(np.float32) - z4)
print("\nINT4 scale:", s4, "zero-point:", z4)
print("INT4 quantised:\n", Wq4)
print("INT4 error (max abs):", np.max(np.abs(Wdeq4 - W)))

# --- Matrix-vector products ---
y_fp32 = W @ x
y_int8 = Wdeq8 @ x
y_int4 = Wdeq4 @ x
print("\nFP32 output:", y_fp32)
print("INT8 output:", y_int8, "  error:", y_int8 - y_fp32)
print("INT4 output:", y_int4, "  error:", y_int4 - y_fp32)
```

**Expected output (abbreviated):**

```
INT8 scale: 0.022362
INT8 error (max abs): 0.01094

INT4 scale: 0.33133  zero-point: 9
INT4 error (max abs): 0.16133

FP32 output: [ 1.854 -6.714  2.631]
INT8 output: [ 1.867 -6.699  2.640]   error: [ 0.013  0.015  0.009]
INT4 output: [ 1.590 -7.124  2.518]   error: [-0.264 -0.410 -0.113]
```
