"""Pure PyTorch reference implementation of the deployed balance-bot policy.

Mirrors rsl_rl's stock feed-forward `ActorCritic` actor MLP exactly: three
`Linear + LeakyReLU` hidden layers followed by a plain `Linear` mean head
(no recurrent state, unlike the RoboCup kick policy, and no tanh output --
`rl_cfg.py`'s `distribution_cfg` uses a plain `GaussianDistribution`, so the
action is clamped by the actuator's MuJoCo `ctrlrange` / the real motor's
duty-cycle clamp instead of a squashing activation).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tools.quantize_export_balance_bot.network_spec import HIDDEN_DIMS, INPUT_DIM, OUTPUT_DIM

LEAKY_RELU_NEGATIVE_SLOPE = 0.01
"""PyTorch/rsl_rl "lrelu" default slope -- verify against the installed
rsl_rl version's `ActorCritic` activation construction if this changes."""


class BalanceBotPolicyRef(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    dims = (INPUT_DIM, *HIDDEN_DIMS)
    self.fc1 = nn.Linear(dims[0], dims[1])
    self.fc2 = nn.Linear(dims[1], dims[2])
    self.fc3 = nn.Linear(dims[2], dims[3])
    self.fc_out = nn.Linear(dims[3], OUTPUT_DIM)
    self.act = nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    x = self.act(self.fc1(obs))
    x = self.act(self.fc2(x))
    x = self.act(self.fc3(x))
    return self.fc_out(x)  # Raw mean action; caller clamps to [-1, 1].


def load_from_rsl_rl_state_dict(
  model: BalanceBotPolicyRef, state_dict: dict, prefix: str = "actor."
) -> None:
  """Copy weights from an rsl_rl `ActorCritic` checkpoint's actor MLP.

  ASSUMPTION / TODO: rsl_rl's stock MLP actor is an `nn.Sequential` of
  alternating `Linear`/`LeakyReLU` layers, so weights are expected at the
  even indices `{prefix}0`, `{prefix}2`, `{prefix}4`, `{prefix}6`. Verify
  this against the installed rsl_rl version's `ActorCritic` before relying
  on it -- adjust the indices below if the layout differs.
  """
  layers = [model.fc1, model.fc2, model.fc3, model.fc_out]
  for i, layer in enumerate(layers):
    idx = i * 2
    layer.weight.data = state_dict[f"{prefix}{idx}.weight"].detach().clone()
    layer.bias.data = state_dict[f"{prefix}{idx}.bias"].detach().clone()


def fold_input_normalization(
  model: BalanceBotPolicyRef, mean: torch.Tensor, std: torch.Tensor
) -> None:
  """Fold `(obs - mean) / std` into `fc1`'s weight/bias so the exported
  network needs no separate normalization step on-device.

  `mean`/`std` come from rsl_rl's `obs_normalization=True` running
  normalizer (see `rl_cfg.py`); their checkpoint key spelling is
  rsl_rl-version specific, so callers must resolve and pass the tensors in
  explicitly rather than this function guessing the key.
  """
  std = std.clamp_min(1e-6)
  with torch.no_grad():
    new_weight = model.fc1.weight / std
    new_bias = model.fc1.bias - model.fc1.weight @ (mean / std)
    model.fc1.weight.copy_(new_weight)
    model.fc1.bias.copy_(new_bias)
