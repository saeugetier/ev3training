"""CLI: checkpoint + rollout -> calibrated Q15 Rust weight file for the
balance-bot policy.

Usage:
    python -m tools.quantize_export_balance_bot.cli \\
        --checkpoint path/to/rsl_rl_checkpoint.pt \\
        --rollout path/to/rollout_obs.npy \\
        --out path/to/policy_weights.rs

`rollout_obs.npy` is a `[T, INPUT_DIM]` float32 array of raw (pre-
normalization) actor observations recorded from `uv run play` on the trained
checkpoint (see `record_rollout.py`). `checkpoint` is adapted into this
repo's `BalanceBotPolicyRef` via
`reference_model.load_from_rsl_rl_state_dict` -- see that function's
docstring for the assumed rsl_rl `ActorCritic` layout.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from tools.quantize_export_balance_bot.calibrate import calibrate
from tools.quantize_export_balance_bot.export_rust import export_policy_weights_rs
from tools.quantize_export_balance_bot.reference_model import (
  BalanceBotPolicyRef,
  fold_input_normalization,
  load_from_rsl_rl_state_dict,
)

# Known rsl_rl checkpoint key spellings for the obs normalizer's running
# mean/var; try each in order since this varies across rsl_rl versions.
_NORMALIZER_KEY_CANDIDATES = (
  ("actor_obs_normalizer.mean", "actor_obs_normalizer._std"),
  ("obs_normalizer.mean", "obs_normalizer._std"),
  ("actor_obs_normalizer.mean", "actor_obs_normalizer.std"),
  ("obs_normalizer.mean", "obs_normalizer.std"),
  ("actor_obs_normalizer.mean", "actor_obs_normalizer.var"),
  ("obs_normalizer.mean", "obs_normalizer.var"),
)


def _normalizer_std(state_dict: dict, std_key: str) -> torch.Tensor:
  std = state_dict[std_key].detach().float().clamp_min(1e-8)
  return std if std_key.endswith(("._std", ".std")) else std.sqrt()


def _mapping_sources(value: object):
  if not isinstance(value, Mapping):
    return
  yield value
  for nested in value.values():
    yield from _mapping_sources(nested)


def _find_normalizer(state: object, state_dict: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
  sources = list(_mapping_sources(state_dict))
  if state is not state_dict:
    sources.extend(_mapping_sources(state))
  for source in sources:
    normalized = {str(key).replace(" ", ""): value for key, value in source.items()}
    for mean_key, std_key in _NORMALIZER_KEY_CANDIDATES:
      if mean_key in normalized and std_key in normalized:
        mean = normalized[mean_key].detach().float()
        return mean, _normalizer_std(normalized, std_key)
  return None


def _normalizer_keys(state: object) -> list[str]:
  return sorted(
    {
      str(key)
      for source in _mapping_sources(state)
      for key in source
      if "normalizer" in str(key).lower()
    }
  )


def _extract_state_dict(state: object) -> dict:
  """Extract model weights from common rsl_rl/PyTorch checkpoint layouts."""
  if not isinstance(state, dict):
    raise TypeError(
      f"Checkpoint must contain a mapping, got {type(state).__name__}."
    )

  for key in ("model_state_dict", "state_dict", "actor_state_dict"):
    candidate = state.get(key)
    if isinstance(candidate, dict):
      return candidate

  if any(key.endswith(".weight") or key.endswith(".bias") for key in state):
    return state

  available = ", ".join(str(key) for key in state.keys())
  raise KeyError(
    "Could not find model weights in checkpoint. Expected one of "
    "'model_state_dict', 'state_dict', or 'actor_state_dict', or a direct "
    f"state dict. Available top-level keys: {available}"
  )


def _resolve_actor_prefix(state_dict: dict, requested: str | None = None) -> str:
  """Resolve the layer-key prefix for a full or actor-only state dict."""
  if requested is not None:
    return requested
  if "mlp.0.weight" in state_dict:
    return "mlp."
  if "actor.0.weight" in state_dict:
    return "actor."
  if "0.weight" in state_dict:
    return ""
  available = ", ".join(str(key) for key in list(state_dict.keys())[:12])
  raise KeyError(
    "Could not find the first actor layer. Expected 'mlp.0.weight', "
    "'actor.0.weight', or '0.weight'; "
    f"sample state-dict keys: {available}"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--rollout", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--actor-prefix",
    default=None,
    help="Optional state-dict key prefix; auto-detected by default.",
  )
  args = parser.parse_args()

  model = BalanceBotPolicyRef()
  state = torch.load(args.checkpoint, map_location="cpu")
  state_dict = _extract_state_dict(state)
  actor_prefix = _resolve_actor_prefix(state_dict, args.actor_prefix)
  load_from_rsl_rl_state_dict(model, state_dict, prefix=actor_prefix)

  # rl_cfg.py sets obs_normalization=True -- fold the running normalizer
  # into fc1 if present under a known key spelling, otherwise assume the
  # rollout was already recorded post-normalization.
  normalizer = _find_normalizer(state, state_dict)
  if normalizer is not None:
    fold_input_normalization(model, *normalizer)
  else:
    print(
      "WARNING: no obs normalizer found in checkpoint under the known key "
      "spellings; assuming the rollout observations are already normalized, "
      "or that the policy was trained without normalization. "
      f"Normalizer-like keys: {_normalizer_keys(state)}"
    )

  model.eval()

  obs_sequence = np.load(args.rollout).astype(np.float32)
  stats = calibrate(model, obs_sequence)

  rust_src = export_policy_weights_rs(model, stats)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(rust_src)
  print(f"Wrote {args.out} ({len(rust_src)} bytes)")


if __name__ == "__main__":
  main()
