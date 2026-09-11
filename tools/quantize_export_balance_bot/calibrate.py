"""Activation range calibration for Q15 export of the balance-bot policy.

Unlike the RoboCup kick policy's LSTM gates, every intermediate tensor here
is a plain FC pre-activation logit (no sigmoid/tanh), so calibration only
needs the observed absmax of the raw observation and each hidden layer's
pre-activation logit, gathered by replaying a recorded rollout of raw
(pre-normalization) observations through the reference model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tools.quantize_export_balance_bot.reference_model import BalanceBotPolicyRef


@dataclass
class CalibrationStats:
  obs_absmax: float = 1e-8
  fc1_absmax: float = 1e-8
  fc2_absmax: float = 1e-8
  fc3_absmax: float = 1e-8
  action_absmax: float = 1.0  # Never shrink below the [-1, 1] actuator range.


@torch.no_grad()
def calibrate(model: BalanceBotPolicyRef, obs_sequence: np.ndarray) -> CalibrationStats:
  """`obs_sequence`: `[T, INPUT_DIM]` float32 array of raw observations from
  a recorded rollout (post `fold_input_normalization` if that was applied to
  `model`, i.e. the same observations the real robot would feed in)."""
  stats = CalibrationStats()
  obs = torch.from_numpy(obs_sequence).float()
  stats.obs_absmax = max(stats.obs_absmax, obs.abs().max().item())

  logit1 = model.fc1(obs)
  stats.fc1_absmax = max(stats.fc1_absmax, logit1.abs().max().item())
  x1 = model.act(logit1)

  logit2 = model.fc2(x1)
  stats.fc2_absmax = max(stats.fc2_absmax, logit2.abs().max().item())
  x2 = model.act(logit2)

  logit3 = model.fc3(x2)
  stats.fc3_absmax = max(stats.fc3_absmax, logit3.abs().max().item())
  x3 = model.act(logit3)

  action = model.fc_out(x3)
  stats.action_absmax = max(stats.action_absmax, action.abs().max().item())

  return stats
