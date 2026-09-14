"""Compare PyTorch and Rust-policy inputs/outputs for one Q15 observation.

The input must be exactly the 36 int16 values printed or accepted by the
Rust `policy_probe`. The values are dequantized with the OBS_SCALE emitted in
`firmware_balance_bot/src/policy_weights.rs`, then evaluated by the checkpoint
model after folding its observation normalizer into the first layer.

Example:
    uv run --project balance_robot_rl python \
      -m tools.quantize_export_balance_bot.compare_policy \
      --checkpoint path/to/model_2500.pt \
      --input 0 0 0 ... 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from tools.quantize_export_balance_bot.cli import (
  _extract_state_dict,
  _resolve_actor_prefix,
  _NORMALIZER_KEY_CANDIDATES,
)
from tools.quantize_export_balance_bot.network_spec import INPUT_DIM
from tools.quantize_export_balance_bot.reference_model import (
  BalanceBotPolicyRef,
  fold_input_normalization,
  load_from_rsl_rl_state_dict,
)


def read_rust_scale(path: Path, name: str) -> float:
  text = path.read_text()
  match = re.search(rf"pub const {name}: f32 = ([^;]+);", text)
  if match is None:
    raise ValueError(f"Could not find {name} in {path}")
  return float(match.group(1))


def load_reference(checkpoint: Path) -> BalanceBotPolicyRef:
  model = BalanceBotPolicyRef()
  state = torch.load(checkpoint, map_location="cpu")
  state_dict = _extract_state_dict(state)
  prefix = _resolve_actor_prefix(state_dict)
  load_from_rsl_rl_state_dict(model, state_dict, prefix=prefix)

  for mean_key, var_key in _NORMALIZER_KEY_CANDIDATES:
    if mean_key in state_dict and var_key in state_dict:
      mean = state_dict[mean_key].detach().float()
      std = state_dict[var_key].detach().float().clamp_min(1e-8).sqrt()
      fold_input_normalization(model, mean, std)
      break
  else:
    print("warning: observation normalizer was not found")

  model.eval()
  return model


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--weights-rs",
    type=Path,
    default=Path("firmware_balance_bot/src/policy_weights.rs"),
  )
  parser.add_argument("--input", type=int, nargs="+", required=True)
  args = parser.parse_args()

  if len(args.input) != INPUT_DIM:
    raise SystemExit(f"expected {INPUT_DIM} Q15 values, got {len(args.input)}")

  obs_scale = read_rust_scale(args.weights_rs, "OBS_SCALE")
  action_scale = read_rust_scale(args.weights_rs, "ACTION_SCALE")
  model = load_reference(args.checkpoint)
  obs_q15 = torch.tensor(args.input, dtype=torch.float32)
  with torch.no_grad():
    action = model((obs_q15 * obs_scale).unsqueeze(0))[0]

  print(f"obs_scale={obs_scale:.10g}")
  print(f"action_scale={action_scale:.10g}")
  print(
    "pytorch_action_raw: "
    f"left={action[0].item():.8f} right={action[1].item():.8f}"
  )
  print(
    "pytorch_action_clamped: "
    f"left={action[0].clamp(-1, 1).item():.8f} "
    f"right={action[1].clamp(-1, 1).item():.8f}"
  )
  print(
    "compare_with_rust: run policy_probe with the same 36 integers; "
    "Rust action_real should be close, with quantization error expected."
  )


if __name__ == "__main__":
  main()
