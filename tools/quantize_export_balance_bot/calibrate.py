"""Activation range calibration for Q15 export of the balance-bot policy.

Unlike the RoboCup kick policy's LSTM gates, every intermediate tensor here
is a plain FC pre-activation logit (no sigmoid/tanh), so calibration only
needs the observed range of the raw observation and each hidden layer's
pre-activation logit, gathered by replaying a recorded rollout of raw
(pre-normalization) observations through the reference model.

Uses a high percentile instead of the true max: the achievable resolution
of `fully_connected_s16`'s Q15 pipeline is `32768 * input_scale *
weight_scale`, independent of the *output* scale, so a single rare outlier
sample (e.g. `last_action` spiking during a fall/reset transient) inflating
`input_scale` coarsens the quantization for every other, typical-magnitude
sample -- verified empirically: one such outlier alone matched the entire
rollout's `obs_absmax`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.quantize_export_balance_bot.reference_model import BalanceBotPolicyRef

CALIBRATION_PERCENTILE = 99.9


@dataclass
class CalibrationStats:
  obs_absmax: float = 1e-8
  fc1_absmax: float = 1e-8
  fc2_absmax: float = 1e-8
  fc3_absmax: float = 1e-8
  action_absmax: float = 1.0  # Never shrink below the [-1, 1] actuator range.


def _robust_absmax(values: torch.Tensor, percentile: float = CALIBRATION_PERCENTILE) -> float:
  """High-percentile absmax: robust to the rare outlier sample that would
  otherwise single-handedly set (and coarsen) the whole quantization scale.
  """
  return torch.quantile(values.abs().flatten(), percentile / 100.0).item()


@torch.no_grad()
def calibrate(model: BalanceBotPolicyRef, obs_sequence: np.ndarray) -> CalibrationStats:
  """`obs_sequence`: `[T, INPUT_DIM]` float32 array of raw observations from
  a recorded rollout (post `fold_input_normalization` if that was applied to
  `model`, i.e. the same observations the real robot would feed in)."""
  stats = CalibrationStats()
  obs = torch.from_numpy(obs_sequence).float()
  stats.obs_absmax = max(stats.obs_absmax, _robust_absmax(obs))

  logit1 = model.fc1(obs)
  stats.fc1_absmax = max(stats.fc1_absmax, _robust_absmax(logit1))
  x1 = model.act(logit1)

  logit2 = model.fc2(x1)
  stats.fc2_absmax = max(stats.fc2_absmax, _robust_absmax(logit2))
  x2 = model.act(logit2)

  logit3 = model.fc3(x2)
  stats.fc3_absmax = max(stats.fc3_absmax, _robust_absmax(logit3))
  x3 = model.act(logit3)

  action = model.fc_out(x3)
  stats.action_absmax = max(stats.action_absmax, _robust_absmax(action))

  return stats
