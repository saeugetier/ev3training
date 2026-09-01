"""Episode-reset terms for the RoboCup task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from mjlab_robocup.assets.field_asset import (
    BALL_RADIUS_M,
    FIELD_HALF_LENGTH_M,
    FIELD_HALF_WIDTH_M,
)

BALL_ROBOT_MIN_DISTANCE_M = 0.40
FIELD_MARGIN_M = 0.10


def reset_ball_randomly(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
    """Place the ball randomly within the field, away from the robot spawn."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    ball = env.scene["ball"]
    root_states = ball.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    half_length = FIELD_HALF_LENGTH_M - FIELD_MARGIN_M
    half_width = FIELD_HALF_WIDTH_M - FIELD_MARGIN_M
    xy = torch.empty((len(env_ids), 2), device=env.device)
    xy[:, 0].uniform_(-half_length, half_length)
    xy[:, 1].uniform_(-half_width, half_width)

    for _ in range(7):
        invalid = torch.linalg.vector_norm(xy, dim=-1) < BALL_ROBOT_MIN_DISTANCE_M
        if not torch.any(invalid):
            break
        candidates = torch.empty_like(xy)
        candidates[:, 0].uniform_(-half_length, half_length)
        candidates[:, 1].uniform_(-half_width, half_width)
        xy[invalid] = candidates[invalid]

    invalid = torch.linalg.vector_norm(xy, dim=-1) < BALL_ROBOT_MIN_DISTANCE_M
    xy[invalid, 0] = BALL_ROBOT_MIN_DISTANCE_M
    root_states[:, :2] = xy + origins[:, :2]
    root_states[:, 2] = BALL_RADIUS_M + origins[:, 2]
    root_states[:, 7:] = 0.0
    ball.write_root_state_to_sim(root_states, env_ids=env_ids)