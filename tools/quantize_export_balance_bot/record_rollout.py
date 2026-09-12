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

The environment is loaded through mjlab's task registry and the policy is
loaded through the same `MjlabOnPolicyRunner` API used by mjlab's `play`
command. The observation array contains the actor observation stream only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def load_env(task: str, num_envs: int = 1):
  """Create a single-env mjlab task instance for rollout recording."""
  import balance_bot.tasks  # noqa: F401  (registers the task id)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(
    cfg=env_cfg,
    device="cuda:0" if _cuda_available() else "cpu",
  )
  return RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)


def _cuda_available() -> bool:
  import torch

  return torch.cuda.is_available()


def load_policy(task: str, checkpoint: Path, env):
  """Load a trained actor from a local rsl_rl checkpoint file.

  This follows the checkpoint-loading path used by mjlab's `play` command.
  """
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  device = str(env.device)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def record_rollout(task: str, checkpoint: Path, steps: int) -> np.ndarray:
  env = load_env(task)
  policy = load_policy(task, checkpoint, env)

  obs = env.get_observations()
  actor_obs = obs["actor"]
  observations = np.zeros((steps, actor_obs.shape[-1]), dtype=np.float32)
  for t in range(steps):
    observations[t] = obs["actor"].detach().cpu().numpy()[0]
    with torch.no_grad():
      action = policy(obs)
    obs, _reward, dones, _info = env.step(action)
    if bool(dones[0]):
      obs = env.get_observations()
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
