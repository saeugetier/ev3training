"""Symmetric int8-weight / int16-activation quantization utilities, tuned to
embedded-nn's actual `fully_connected_s16` semantics (verified empirically
against the installed crate for the RoboCup kick policy -- see
`tools/quantize_export_robocup/quantize.py` and
`/memories/embedded-nn-crate-facts.md`). Duplicated here (rather than
imported cross-package) since each deployed task in this repo keeps its own
self-contained export pipeline.

- Weights are int8, bias is int64 (NOT int16/int32 as the generic
  TFLite/CMSIS-NN convention might suggest).
- The kernel's internal `(acc_i64 >> 15) as i32` step before requantization
  means the multiplier must be computed from
  `real_multiplier = (input_scale * weight_scale / output_scale) * 32768`.
"""

from __future__ import annotations

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
  """Biases are quantized to int64, matching `fully_connected_s16`'s
  `bias: Option<&[i64]>` parameter.

  Verified against the installed crate (`fully_connected.rs`): bias is added
  to the accumulator *before* its internal `acc >> 15` step, exactly like the
  weight dot-product terms. `fc_s16_requant_params`'s multiplier already
  compensates for that shift with an extra `*32768` factor on the dot
  product; the bias needs the same `/32768` scale correction, or small (but
  real) bias values get truncated to 0 by the `>>15` before requantization
  ever sees them.
  """
  bias_scale = input_scale * weight_scale / 32768.0
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
  module docstring).
  """
  real_multiplier = (input_scale * weight_scale / output_scale) * 32768.0
  return quantize_multiplier(real_multiplier)
