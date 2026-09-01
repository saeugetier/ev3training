"""Reward terms for the 1v0 dribble-to-goal-and-kick RoboCup task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_robocup.assets.field_asset import ROBOT_WALL_CLEARANCE_M

# Field/goal layout constants (meters), must match the scene MJCF/config.
GOAL_POS_XY = (2.0, 0.0)
GOAL_HALF_WIDTH_M = 0.35
FIELD_HALF_LENGTH_M = 2.2
FIELD_HALF_WIDTH_M = 1.5


def _root_pos_xy(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
    """World-frame root position of a free-floating entity, shape [num_envs, 2]."""
    return env.scene[entity_name].data.root_link_pos_w[:, :2]


def robot_ball_vec(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged critic observation: ball position relative to the robot, [N, 2]."""
    return _root_pos_xy(env, "ball") - _root_pos_xy(env, "robot")


def robot_ball_approach(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Dense reward for closing the distance between robot and ball."""
    robot_xy = _root_pos_xy(env, "robot")
    ball_xy = _root_pos_xy(env, "ball")
    dist = torch.norm(robot_xy - ball_xy, dim=-1)
    return -dist


def ball_goal_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Dense reward for moving the ball toward the opponent goal."""
    ball_xy = _root_pos_xy(env, "ball")
    goal = torch.tensor(GOAL_POS_XY, device=ball_xy.device)
    dist = torch.norm(ball_xy - goal, dim=-1)
    return -dist


def goal_scored_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Sparse bonus the step the ball enters the goal region."""
    ball_xy = _root_pos_xy(env, "ball")
    in_goal_x = ball_xy[:, 0] > GOAL_POS_XY[0]
    in_goal_y = torch.abs(ball_xy[:, 1] - GOAL_POS_XY[1]) < GOAL_HALF_WIDTH_M
    return (in_goal_x & in_goal_y).float()


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize large step-to-step action changes (smoothness, sim2real)."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    return torch.sum((action - prev_action) ** 2, dim=-1)


def wheel_energy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize large commanded wheel speeds (encourages efficient motion)."""
    action = env.action_manager.action
    return torch.sum(action[:, :2] ** 2, dim=-1)


def robot_wall_proximity(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize entering the wall-clearance zone without ending the episode."""
    robot_xy = _root_pos_xy(env, "robot")
    distance_to_wall = torch.minimum(
        FIELD_HALF_LENGTH_M - torch.abs(robot_xy[:, 0]),
        FIELD_HALF_WIDTH_M - torch.abs(robot_xy[:, 1]),
    )
    proximity = (ROBOT_WALL_CLEARANCE_M - distance_to_wall).clamp_min(0.0)
    return (proximity / ROBOT_WALL_CLEARANCE_M) ** 2
