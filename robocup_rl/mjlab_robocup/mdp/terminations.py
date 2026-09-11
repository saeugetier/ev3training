"""Termination terms for the RoboCup task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_robocup.mdp.rewards import (
    FIELD_HALF_LENGTH_M,
    FIELD_HALF_WIDTH_M,
    GOAL_HALF_WIDTH_M,
    GOAL_POS_XY,
    _root_pos_xy,
)


def goal_scored(env: ManagerBasedRlEnv) -> torch.Tensor:
    """End the episode once the ball enters the goal."""
    ball_xy = _root_pos_xy(env, "ball")
    in_goal_x = ball_xy[:, 0] > GOAL_POS_XY[0]
    in_goal_y = torch.abs(ball_xy[:, 1] - GOAL_POS_XY[1]) < GOAL_HALF_WIDTH_M
    return in_goal_x & in_goal_y


def out_of_field_bounds(env: ManagerBasedRlEnv) -> torch.Tensor:
    """End the episode if the ball escapes the physical field boundary."""
    ball_xy = _root_pos_xy(env, "ball")
    ball_out = (torch.abs(ball_xy[:, 0]) > FIELD_HALF_LENGTH_M) | (
        torch.abs(ball_xy[:, 1]) > FIELD_HALF_WIDTH_M
    )
    return ball_out
