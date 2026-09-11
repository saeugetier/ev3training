"""MDP terms specific to the balance bot task.

Sensor model (mirrors the real robot):
  * IMU / gyro: body tilt angle (zeroed at power-up) and three-axis angular rate.
  * Wheel encoders: rotation distance (accumulated angle) of each wheel.
  * Remote control: one continuous signed control value per wheel [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Remote control command.
##


class RemoteControlCommand(CommandTerm):
  """Continuous remote control with independent wheel commands.

  The normalized wheel controls are the only part of the command the policy
  sees; the term additionally integrates the commanded wheel rotation so the
  task can reward odometry tracking.
  """

  cfg: RemoteControlCommandCfg

  def __init__(self, cfg: RemoteControlCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._env = env
    self._asset: Entity = env.scene[cfg.entity_name]
    self._wheel_ids = [
      self._asset.find_joints([name])[0][0] for name in cfg.wheel_joint_names
    ]

    shape = (self.num_envs, 2)
    self._controls = torch.zeros(shape, device=self.device)
    self._ref_wheel_pos = torch.zeros(shape, device=self.device)

    self.metrics["wheel_speed_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["odometry_error"] = torch.zeros(self.num_envs, device=self.device)

  # Properties.

  @property
  def command(self) -> torch.Tensor:
    """Continuous normalized wheel controls for the left and right wheels."""
    return self._controls

  @property
  def target_wheel_vel(self) -> torch.Tensor:
    """Commanded wheel angular velocity [rad/s], shape [num_envs, 2]."""
    return self._controls * self.cfg.wheel_speed

  @property
  def measured_wheel_pos(self) -> torch.Tensor:
    return self._asset.data.joint_pos[:, self._wheel_ids]

  @property
  def measured_wheel_vel(self) -> torch.Tensor:
    return self._asset.data.joint_vel[:, self._wheel_ids]

  @property
  def wheel_pos_error(self) -> torch.Tensor:
    """Commanded minus measured wheel rotation [rad], saturated."""
    error = self._ref_wheel_pos - self.measured_wheel_pos
    return error.clamp(-self.cfg.max_pos_error, self.cfg.max_pos_error)

  # Implementation.

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._controls[env_ids] = torch.empty(
      (len(env_ids), 2), device=self.device
    ).uniform_(*self.cfg.control_range)
    # Start the odometry reference at the current wheel position so a new
    # command never inherits the error accumulated under the previous one.
    self._ref_wheel_pos[env_ids] = self.measured_wheel_pos[env_ids]

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._ref_wheel_pos[ids] += self.target_wheel_vel[ids] * self._env.step_dt
    # Keep the integrator from winding up while the robot is blocked.
    self._ref_wheel_pos[ids] = (
      self.measured_wheel_pos[ids]
      + (self._ref_wheel_pos[ids] - self.measured_wheel_pos[ids]).clamp(
        -self.cfg.max_pos_error, self.cfg.max_pos_error
      )
    )

  def _update_metrics(self) -> None:
    self.metrics["wheel_speed_error"] = torch.norm(
      self.target_wheel_vel - self.measured_wheel_vel, dim=-1
    )
    self.metrics["odometry_error"] = torch.norm(self.wheel_pos_error, dim=-1)


@dataclass(kw_only=True)
class RemoteControlCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  wheel_joint_names: tuple[str, str] = ("left_wheel", "right_wheel")
  wheel_speed: float = 12.0
  """Maximum wheel angular velocity commanded by control 1 [rad/s]."""
  max_pos_error: float = 2.0
  """Saturation of the odometry error [rad of wheel rotation]."""
  control_range: tuple[float, float] = (-1.0, 1.0)
  """Inclusive range for independently sampled normalized wheel controls."""

  def build(self, env: ManagerBasedRlEnv) -> RemoteControlCommand:
    return RemoteControlCommand(self, env)


##
# Observations.
##


def imu_pitch(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Tilt angle about the wheel axis [rad], shape [num_envs, 1].

  Derived from the IMU the same way the real robot's complementary filter does.
  The per-episode bias configured on this observation term models the fact that
  the on-board angle is zeroed once at start-up.
  """
  asset: Entity = env.scene[asset_cfg.name]
  gravity = asset.data.projected_gravity_b
  return torch.atan2(gravity[:, 0], -gravity[:, 2]).unsqueeze(-1)


def wheel_pos_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Commanded minus measured wheel rotation [rad], shape [num_envs, 2]."""
  return env.command_manager.get_term(command_name).wheel_pos_error


def wheel_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Wheel angular velocity [rad/s], as differentiated from the encoders."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


##
# Rewards.
##


def upright(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for keeping the chassis vertical."""
  asset: Entity = env.scene[asset_cfg.name]
  cos_tilt = -asset.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
  tilt = torch.acos(cos_tilt)
  return torch.exp(-torch.square(tilt / std))


def track_wheel_speed(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Reward for matching the wheel speeds requested by the remote control."""
  command = env.command_manager.get_term(command_name)
  error = torch.sum(
    torch.square(command.target_wheel_vel - command.measured_wheel_vel), dim=-1
  )
  return torch.exp(-error / (std**2))


def track_wheel_position(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """Reward for tracking the commanded travelled distance of each wheel.

  With no button pressed this is a station-keeping reward: the robot has to
  balance without drifting away.
  """
  command = env.command_manager.get_term(command_name)
  error = torch.sum(torch.square(command.wheel_pos_error), dim=-1)
  return torch.exp(-error / (std**2))


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation of the axle height, i.e. leaning onto the chassis."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


##
# Events.
##


def _uniform(low: float, high: float, n: int, device: torch.device) -> torch.Tensor:
  return torch.empty(n, device=device).uniform_(low, high)


def throw_box(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  box_names: tuple[str, ...],
  speed_range: tuple[float, float] = (1.5, 3.5),
  distance_range: tuple[float, float] = (0.6, 1.2),
  height_range: tuple[float, float] = (0.1, 0.4),
  spin_range: tuple[float, float] = (-8.0, 8.0),
  aim_height: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Throw one of the light boxes at the robot from a random direction.

  The box is teleported to a random point around the robot and given a ballistic
  velocity that makes it hit the chassis.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return

  device = env.device
  n = len(env_ids)
  robot: Entity = env.scene[asset_cfg.name]

  target = robot.data.root_link_pos_w[env_ids].clone()
  target[:, 2] += aim_height

  angle = _uniform(0.0, 2.0 * torch.pi, n, device)
  distance = _uniform(*distance_range, n, device)
  height = _uniform(*height_range, n, device)
  spawn = torch.stack(
    [
      target[:, 0] + distance * torch.cos(angle),
      target[:, 1] + distance * torch.sin(angle),
      target[:, 2] + height,
    ],
    dim=-1,
  )

  delta = target - spawn
  speed = _uniform(*speed_range, n, device)
  flight_time = delta.norm(dim=-1) / speed
  velocity = delta / flight_time.unsqueeze(-1)
  velocity[:, 2] += 0.5 * 9.81 * flight_time  # Ballistic aim compensation.

  choice = torch.randint(0, len(box_names), (n,), device=device)
  for index, name in enumerate(box_names):
    mask = choice == index
    if not bool(mask.any()):
      continue
    ids = env_ids[mask]
    state = torch.zeros(len(ids), 13, device=device)
    state[:, 0:3] = spawn[mask]
    state[:, 3] = 1.0
    state[:, 7:10] = velocity[mask]
    state[:, 10:13] = torch.empty(len(ids), 3, device=device).uniform_(*spin_range)
    env.scene[name].write_root_state_to_sim(state, env_ids=ids)
