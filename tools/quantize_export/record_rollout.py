"""Record a rollout of actor observations from a trained mjlab checkpoint,
for use as calibration data by `tools/quantize_export/cli.py`.

Usage:
    python -m tools.quantize_export.record_rollout \\
        --task Mjlab-RoboCup-Kick-v0 \\
        --wandb-run-path your-org/mjlab/run-id \\
        --steps 2000 \\
        --out rollout_obs.npy

ASSUMPTION / TODO: this drives the real mjlab `gym.make(task)` env and an
rsl_rl `OnPolicyRunner`-loaded policy, whose exact loading API (fetching a
checkpoint from a wandb run path, extracting the actor module, resetting
recurrent hidden state) is not yet verified against an installed mjlab/
rsl_rl version -- see /memories/session/plan.md. Adjust the
`load_policy`/`load_env` functions below once mjlab is actually installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_env(task: str, num_envs: int = 1):
    """Create a single-env mjlab task instance for rollout recording."""
    import gymnasium as gym

    import mjlab_robocup  # noqa: F401  (registers the task id)

    env = gym.make(task, env_cfg_overrides={"scene": {"num_envs": num_envs}})
    return env


def load_policy(wandb_run_path: str):
    """Load a trained actor (with recurrent state) from an rsl_rl checkpoint
    fetched via its mjlab/wandb run path.

    NOTE: rsl_rl's `OnPolicyRunner.load(...)` + `.get_inference_policy(...)`
    is the typical entry point for this in Isaac-Lab-family projects;
    verify the exact call against your installed rsl_rl/mjlab version.
    """
    raise NotImplementedError(
        "Wire this up to rsl_rl's checkpoint loading once mjlab/rsl_rl are "
        "installed in this environment -- see this module's docstring."
    )


def record_rollout(task: str, wandb_run_path: str, steps: int) -> np.ndarray:
    env = load_env(task)
    policy = load_policy(wandb_run_path)

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
    parser.add_argument("--task", default="Mjlab-RoboCup-Kick-v0")
    parser.add_argument("--wandb-run-path", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    observations = record_rollout(args.task, args.wandb_run_path, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, observations)
    print(f"Wrote {args.out} ({observations.shape})")


if __name__ == "__main__":
    main()
