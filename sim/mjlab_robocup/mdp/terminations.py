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
from mjlab_robocup.assets.field_asset import ROBOT_WALL_CLEARANCE_M


def goal_scored(env: ManagerBasedRlEnv) -> torch.Tensor:
    """End the episode once the ball enters the goal."""
    ball_xy = _root_pos_xy(env, "ball")
    in_goal_x = ball_xy[:, 0] > GOAL_POS_XY[0]
    in_goal_y = torch.abs(ball_xy[:, 1] - GOAL_POS_XY[1]) < GOAL_HALF_WIDTH_M
    return in_goal_x & in_goal_y


def out_of_field_bounds(env: ManagerBasedRlEnv) -> torch.Tensor:
    """End the episode before the robot can reach a perimeter wall."""
    ball_xy = _root_pos_xy(env, "ball")
    robot_xy = _root_pos_xy(env, "robot")
    ball_out = (torch.abs(ball_xy[:, 0]) > FIELD_HALF_LENGTH_M) | (
        torch.abs(ball_xy[:, 1]) > FIELD_HALF_WIDTH_M
    )
    robot_out = (
        torch.abs(robot_xy[:, 0]) > FIELD_HALF_LENGTH_M - ROBOT_WALL_CLEARANCE_M
    ) | (
        torch.abs(robot_xy[:, 1]) > FIELD_HALF_WIDTH_M - ROBOT_WALL_CLEARANCE_M
    )
    return ball_out | robot_out
