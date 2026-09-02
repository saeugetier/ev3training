"""Reward terms for the 1v0 dribble-to-goal-and-kick RoboCup task."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab_robocup.assets.field_asset import BALL_RADIUS_M, ROBOT_WALL_CLEARANCE_M
from mjlab_robocup.mdp.actions import KICK_TRIGGER_THRESHOLD

# Field/goal layout constants (meters), must match the scene MJCF/config.
GOAL_POS_XY = (2.0, 0.0)
GOAL_HALF_WIDTH_M = 0.35
FIELD_HALF_LENGTH_M = 2.2
FIELD_HALF_WIDTH_M = 1.5

# Kicker geometry (chassis-forward offset to the arm tip), see
# robot_asset.py's kicker_arm/kicker_geom placement.
KICKER_REACH_M = 0.14
KICK_CONTACT_MARGIN_M = 0.05
BALL_KICKER_CONTACT_DIST_M = KICKER_REACH_M + BALL_RADIUS_M + KICK_CONTACT_MARGIN_M


def _root_pos_xy(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
    """World-frame root position of a free-floating entity, shape [num_envs, 2]."""
    return env.scene[entity_name].data.root_link_pos_w[:, :2]


def robot_ball_vec(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Privileged critic observation: ball position relative to the robot, [N, 2]."""
    return _root_pos_xy(env, "ball") - _root_pos_xy(env, "robot")


class _distance_progress:
    """Reward the per-step reduction in a distance, independently per env."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._previous_distance = torch.full(
            (env.num_envs,), float("nan"), device=env.device
        )

    def _distance(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        distance = self._distance(env)
        progress = self._previous_distance - distance
        self._previous_distance.copy_(distance)
        return torch.where(torch.isnan(progress), torch.zeros_like(progress), progress)

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        self._previous_distance[env_ids] = float("nan")


class robot_ball_approach(_distance_progress):
    """Reward only movement that reduces robot-to-ball distance."""

    def _distance(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return torch.norm(robot_ball_vec(env), dim=-1)


class ball_goal_progress(_distance_progress):
    """Reward only movement that reduces ball-to-goal distance."""

    def _distance(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        ball_xy = _root_pos_xy(env, "ball")
        goal = torch.tensor(GOAL_POS_XY, device=ball_xy.device)
        return torch.norm(ball_xy - goal, dim=-1)


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


def _ball_kicker_distance(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Distance from the ball to the kicker arm tip, shape [num_envs]."""
    robot = env.scene["robot"]
    heading = robot.data.heading_w
    forward_xy = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    kicker_xy = _root_pos_xy(env, "robot") + forward_xy * KICKER_REACH_M
    ball_xy = _root_pos_xy(env, "ball")
    return torch.norm(ball_xy - kicker_xy, dim=-1)


def kick_misuse_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize firing the kicker while the ball is out of contact range."""
    kick_active = (env.action_manager.action[:, 2] > KICK_TRIGGER_THRESHOLD).float()
    ball_out_of_range = (_ball_kicker_distance(env) > BALL_KICKER_CONTACT_DIST_M).float()
    return kick_active * ball_out_of_range


def ball_possession(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Reward the ball being close and in front of the kicker."""
    robot = env.scene["robot"]
    heading = robot.data.heading_w
    forward_xy = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    ball_from_kicker = _root_pos_xy(env, "ball") - (
        _root_pos_xy(env, "robot") + forward_xy * KICKER_REACH_M
    )
    forward_distance = torch.sum(ball_from_kicker * forward_xy, dim=-1)
    lateral_distance = torch.abs(
        ball_from_kicker[:, 0] * forward_xy[:, 1]
        - ball_from_kicker[:, 1] * forward_xy[:, 0]
    )
    is_controlled = (forward_distance > 0.0) & (forward_distance < 0.20)
    return is_controlled.float() * torch.exp(-25.0 * lateral_distance.square())


def robot_wall_proximity(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize entering the wall-clearance zone without ending the episode."""
    robot_xy = _root_pos_xy(env, "robot")
    distance_to_wall = torch.minimum(
        FIELD_HALF_LENGTH_M - torch.abs(robot_xy[:, 0]),
        FIELD_HALF_WIDTH_M - torch.abs(robot_xy[:, 1]),
    )
    proximity = (ROBOT_WALL_CLEARANCE_M - distance_to_wall).clamp_min(0.0)
    return (proximity / ROBOT_WALL_CLEARANCE_M) ** 2
