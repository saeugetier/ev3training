"""Symmetric int8-weight / int16-activation quantization utilities, tuned to
embedded-nn's actual `fully_connected_s16` semantics (verified empirically
against the installed crate, not just its docs -- see
/memories/embedded-nn-crate-facts.md):

- Weights are int8, bias is int64 (NOT int16/int32 as the generic
  TFLite/CMSIS-NN convention might suggest).
- The kernel's internal `(acc_i64 >> 15) as i32` step before requantization
  means the multiplier must be computed from
  `real_multiplier = (input_scale * weight_scale / output_scale) * 32768`.
- `sigmoid_s16`/`tanh_s16(_, _, left_shift)` expect input in Q3.12 fixed
  point at `left_shift=0` (`input_scale = 1/(4096 * 2**left_shift)`) and
  produce Q0.15 output (`real = raw / 32768.0`).
"""

from __future__ import annotations

import math

import numpy as np

INT8_MAX = 127
INT8_MIN = -128
INT16_MAX = 32767
INT16_MIN = -32768


def calibrate_symmetric_scale(values: np.ndarray, num_bits: int = 8) -> float:
    """Compute a symmetric quantization scale from observed min/max."""
    max_abs = float(np.max(np.abs(values))) if values.size else 1.0
    max_abs = max(max_abs, 1e-8)
    q_max = (1 << (num_bits - 1)) - 1
    return max_abs / q_max


def quantize_weight_s8(weight: np.ndarray, scale: float) -> np.ndarray:
    """Quantize a weight tensor to int8, matching `fully_connected_s16`'s
    `kernel: &[i8]` parameter."""
    q = np.round(weight / scale).astype(np.int64)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def quantize_bias_s64(bias: np.ndarray, input_scale: float, weight_scale: float) -> np.ndarray:
    """Biases are quantized to int64 at `input_scale * weight_scale`,
    matching `fully_connected_s16`'s `bias: Option<&[i64]>` parameter."""
    bias_scale = input_scale * weight_scale
    q = np.round(bias / bias_scale).astype(np.int64)
    return q


def quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    """Decompose a real multiplier into (int32 significand, shift), matching
    embedded-nn's `support::requantize` convention: `result ~= val *
    (multiplier / 2**31) * 2**shift` (standard TFLite `QuantizeMultiplier`).
    """
    if real_multiplier == 0.0:
        return 0, 0
    m = real_multiplier
    shift = 0
    while m < 0.5:
        m *= 2.0
        shift -= 1
    while m > 1.0:
        m /= 2.0
        shift += 1
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        shift += 1
    return q, shift


def fc_s16_requant_params(
    input_scale: float, weight_scale: float, output_scale: float
) -> tuple[int, int]:
    """Multiplier/shift for `fully_connected_s16`'s `PerTensorQuantParams`.

    Includes the extra factor of 2**15 to compensate for the kernel's
    internal `(acc_i64 >> 15) as i32` step before requantization (see
    module docstring / memories/embedded-nn-crate-facts.md).
    """
    real_multiplier = (input_scale * weight_scale / output_scale) * 32768.0
    return quantize_multiplier(real_multiplier)


def mul_s16_requant_params(
    input1_scale: float, input2_scale: float, output_scale: float
) -> tuple[int, int]:
    """Multiplier/shift for `elementwise_mul_s16`'s `output_mult`/`output_shift`.

    `elementwise_mul_s16` also does `(prod_i64 >> 15) as i32` internally
    before requantization, same quirk as `fully_connected_s16`.
    """
    real_multiplier = (input1_scale * input2_scale / output_scale) * 32768.0
    return quantize_multiplier(real_multiplier)


def add_s16_input_requant_params(input_scale: float, common_scale: float) -> tuple[int, int]:
    """Multiplier/shift to bring one `elementwise_add_s16` input to a common
    scale before summation (no `>>15` quirk for add, unlike FC/mul)."""
    return quantize_multiplier(input_scale / common_scale)


def add_s16_output_requant_params(common_scale: float, output_scale: float) -> tuple[int, int]:
    """Multiplier/shift for `elementwise_add_s16`'s final output requant."""
    return quantize_multiplier(common_scale / output_scale)


def pow2_activation_scale(
    absmax: float, min_left_shift: int = -8, max_left_shift: int = 8
) -> tuple[float, int]:
    """Pick `(scale, left_shift)` for a tensor that feeds `sigmoid_s16`/
    `tanh_s16`, such that `scale == 1 / (4096 * 2**left_shift)` exactly
    (so the activation's fixed Q3.12 input convention is met with no
    extra rounding error), while keeping `absmax / scale <= 32767`.
    """
    absmax = max(absmax, 1e-8)
    left_shift = math.floor(math.log2(32767.0 / (4096.0 * absmax)))
    left_shift = max(min_left_shift, min(max_left_shift, left_shift))
    scale = 1.0 / (4096.0 * (2.0**left_shift))
    return scale, left_shift
