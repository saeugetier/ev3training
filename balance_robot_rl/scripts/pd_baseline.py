"""Hand-tuned PD balance controller, used as a baseline / model sanity check.

Run with: ``uv run python scripts/pd_baseline.py``
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Drive-BalanceBot"

KP_PITCH = 12.0
KD_PITCH = 1.5
KP_WHEEL = 0.03


def main() -> None:
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 8
  cfg.episode_length_s = 10.0

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  robot = env.scene["robot"]
  env.reset()

  alive = 0.0
  steps = 400
  for _ in range(steps):
    gravity = robot.data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
    pitch_rate = robot.data.root_link_ang_vel_b[:, 1]
    wheel_speed = robot.data.joint_vel.mean(dim=-1)

    torque = (
      KP_PITCH * pitch + KD_PITCH * pitch_rate + KP_WHEEL * wheel_speed
    ).clamp(-1.0, 1.0)
    action = torque.unsqueeze(-1).repeat(1, 2)

    _, _, terminated, _, _ = env.step(action)
    alive += (~terminated).float().mean().item()

  print(f"Alive fraction over {steps} steps: {alive / steps:.3f}")
  env.close()


if __name__ == "__main__":
  main()
