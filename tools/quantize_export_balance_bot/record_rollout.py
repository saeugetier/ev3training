"""Record a rollout of actor observations from a trained mjlab checkpoint,
for use as calibration data by `tools/quantize_export_balance_bot/cli.py`.

Usage:
    python -m tools.quantize_export_balance_bot.record_rollout \\
        --task Mjlab-Drive-BalanceBot \\
        --checkpoint logs/rsl_rl/balance_bot/<run>/model_2500.pt \\
        --steps 2000 \\
        --out rollout_obs.npy

`--checkpoint` is a local `.pt` file path -- rsl_rl's `OnPolicyRunner` always
saves checkpoints to the local `log_dir` regardless of which logger
(`tensorboard`, `wandb`, `neptune`) is selected for metric streaming, so a
tensorboard-only run's checkpoint works here unchanged.

ASSUMPTION / TODO: this drives the real mjlab `gym.make(task)` env and an
rsl_rl `OnPolicyRunner`-loaded policy, whose exact loading API (constructing
an `OnPolicyRunner`, calling `.load(checkpoint)`, extracting the actor via
`.get_inference_policy()`) is not yet verified against an installed
mjlab/rsl_rl version. Adjust the `load_policy`/`load_env` functions below
once mjlab is actually installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_env(task: str, num_envs: int = 1):
  """Create a single-env mjlab task instance for rollout recording."""
  import gymnasium as gym

  import balance_bot.tasks  # noqa: F401  (registers the task id)

  env = gym.make(task, env_cfg_overrides={"scene": {"num_envs": num_envs}})
  return env


def load_policy(checkpoint: Path):
  """Load a trained actor from a local rsl_rl checkpoint file.

  NOTE: rsl_rl's `OnPolicyRunner.load(...)` + `.get_inference_policy(...)`
  is the typical entry point for this in Isaac-Lab-family projects; verify
  the exact call against your installed rsl_rl/mjlab version.
  """
  raise NotImplementedError(
    "Wire this up to rsl_rl's checkpoint loading once mjlab/rsl_rl are "
    "installed in this environment -- see this module's docstring."
  )


def record_rollout(task: str, checkpoint: Path, steps: int) -> np.ndarray:
  env = load_env(task)
  policy = load_policy(checkpoint)

  obs, _ = env.reset()
  observations = np.zeros((steps, obs.shape[-1]), dtype=np.float32)
  for t in range(steps):
    observations[t] = obs.detach().cpu().numpy()[0]
    action = policy(obs)
    obs, _reward, terminated, truncated, _info = env.step(action)
    if terminated[0] or truncated[0]:
      obs, _ = env.reset()
  return observations


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Mjlab-Drive-BalanceBot")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()

  observations = record_rollout(args.task, args.checkpoint, args.steps)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.save(args.out, observations)
  print(f"Wrote {args.out} ({observations.shape})")


if __name__ == "__main__":
  main()
