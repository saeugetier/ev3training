"""Episode-reset terms for the RoboCup task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from mjlab_robocup.assets.field_asset import BALL_RADIUS_M

BALL_START_FORWARD_MIN_M = 0.45
BALL_START_FORWARD_MAX_M = 0.70
BALL_START_LATERAL_MAX_M = 0.15


def reset_ball_randomly(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
    """Place the ball in front of the robot for the initial learning curriculum."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    ball = env.scene["ball"]
    root_states = ball.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    xy = torch.empty((len(env_ids), 2), device=env.device)
    xy[:, 0].uniform_(BALL_START_FORWARD_MIN_M, BALL_START_FORWARD_MAX_M)
    xy[:, 1].uniform_(-BALL_START_LATERAL_MAX_M, BALL_START_LATERAL_MAX_M)
    root_states[:, :2] = xy + origins[:, :2]
    root_states[:, 2] = BALL_RADIUS_M + origins[:, 2]
    root_states[:, 7:] = 0.0
    ball.write_root_state_to_sim(root_states, env_ids=env_ids)