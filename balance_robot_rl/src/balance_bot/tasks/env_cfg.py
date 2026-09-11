"""Environment configuration for the balance bot drive task.

The policy must keep the inverted-pendulum chassis upright while following the
commands of two continuous signed wheel controls in [-1, 1].
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import flat, random_rough, wave_terrain
from mjlab.utils.noise import (
  GaussianNoiseCfg,
  NoiseModelWithAdditiveBiasCfg,
  UniformNoiseCfg,
)
from mjlab.viewer import ViewerConfig

from balance_bot.boxes import BOX_NAMES, get_box_cfgs
from balance_bot.robot import WHEEL_NAMES, get_balance_bot_cfg
from balance_bot.tasks import mdp

ROBOT_CFG = SceneEntityCfg("robot")
WHEELS_CFG = SceneEntityCfg("robot", joint_names=WHEEL_NAMES)
TERRAIN_CFG = SceneEntityCfg("terrain")

COMMAND_NAME = "remote"
WHEEL_SPEED = 12.0
"""Maximum wheel speed commanded by a normalized wheel control of 1 [rad/s]."""

GROUND_FRICTION_RANGE = (0.25, 1.2)
"""Tangential friction of the ground, resampled every episode."""


def uneven_terrain_cfg() -> TerrainGeneratorCfg:
  """Slightly uneven ground: bumps and long waves of a few centimeters."""
  return TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=2.0,
    num_rows=5,
    num_cols=5,
    color_scheme="height",
    add_lights=True,
    sub_terrains={
      "flat": flat(proportion=0.25),
      "bumps": random_rough(
        proportion=0.45,
        noise_range=(0.005, 0.02),
        noise_step=0.005,
        horizontal_scale=0.10,
        border_width=0.25,
      ),
      "waves": wave_terrain(
        proportion=0.3,
        amplitude_range=(0.01, 0.065),
        num_waves=3,
        border_width=0.25,
      ),
    },
  )


def balance_bot_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  ##
  # Commands: signed wheel controls for the left and right wheels.
  ##

  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: mdp.RemoteControlCommandCfg(
      resampling_time_range=(2.0, 5.0),
      entity_name="robot",
      wheel_joint_names=WHEEL_NAMES,
      wheel_speed=WHEEL_SPEED,
      max_pos_error=2.0,
    ),
  }

  ##
  # Observations: exactly what the on-board electronics can measure.
  ##

  # The on-board angle is zeroed once at power-up, which shows up as a constant
  # offset per episode on top of the per-sample gyro noise.
  gyro_angle_noise = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=GaussianNoiseCfg(std=0.005),
    bias_noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
  )

  sensor_terms = {
    "gyro_angle": ObservationTermCfg(
      func=mdp.imu_pitch,
      params={"asset_cfg": ROBOT_CFG},
      noise=gyro_angle_noise,
    ),
    "gyro_rate": ObservationTermCfg(
      func=base_mdp.base_ang_vel,
      params={"asset_cfg": ROBOT_CFG},
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
      scale=0.25,
    ),
    "wheel_speed": ObservationTermCfg(
      func=mdp.wheel_vel,
      params={"asset_cfg": WHEELS_CFG},
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
      clip=(-100.0, 100.0),
      scale=0.1,
    ),
    "wheel_distance_error": ObservationTermCfg(
      func=mdp.wheel_pos_error,
      params={"command_name": COMMAND_NAME},
      scale=0.5,
    ),
    "remote_control": ObservationTermCfg(
      func=base_mdp.generated_commands,
      params={"command_name": COMMAND_NAME},
    ),
    "last_action": ObservationTermCfg(func=base_mdp.last_action),
  }

  # Privileged signals the real robot cannot measure, used by the critic only.
  privileged_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.base_lin_vel, params={"asset_cfg": ROBOT_CFG}
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity, params={"asset_cfg": ROBOT_CFG}
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=dict(sensor_terms),
      enable_corruption=not play,
      history_length=3,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms={**sensor_terms, **privileged_terms},
      enable_corruption=False,
      nan_policy="sanitize",
    ),
  }

  ##
  # Actions: motor effort per wheel, matching the EV3's duty-cycle-only
  # motor control (no direct torque or speed regulation on-device).
  ##

  actions: dict[str, ActionTermCfg] = {
    "wheel_effort": JointEffortActionCfg(
      entity_name="robot",
      actuator_names=WHEEL_NAMES,
      preserve_order=True,
      scale=1.0,
    ),
  }

  ##
  # Rewards.
  ##

  rewards = {
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=2.0,
      params={"std": 0.25, "asset_cfg": ROBOT_CFG},
    ),
    "track_wheel_speed": RewardTermCfg(
      func=mdp.track_wheel_speed,
      weight=1.5,
      params={"command_name": COMMAND_NAME, "std": 4.0},
    ),
    "track_wheel_distance": RewardTermCfg(
      func=mdp.track_wheel_position,
      weight=1.0,
      params={"command_name": COMMAND_NAME, "std": 1.0},
    ),
    "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
    "terminated": RewardTermCfg(func=base_mdp.is_terminated, weight=-10.0),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.01),
    "motor_torque": RewardTermCfg(
      func=base_mdp.joint_torques_l2,
      weight=-0.005,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(WHEEL_NAMES))},
    ),
  }

  ##
  # Events.
  ##

  events = {
    "ground_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={
        "asset_cfg": TERRAIN_CFG,
        "operation": "abs",
        "ranges": GROUND_FRICTION_RANGE,
        "shared_random": True,
      },
    ),
    "reset_scene": EventTermCfg(
      func=base_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "reset_base": EventTermCfg(
      func=base_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # The robot boots up roughly, but not exactly, upright.
        "pose_range": {"z": (0.01, 0.03), "pitch": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
        "velocity_range": {"x": (-0.1, 0.1), "pitch": (-0.2, 0.2)},
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "reset_wheels": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": WHEELS_CFG,
      },
    ),
    "push_robot": EventTermCfg(
      func=base_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 8.0),
      params={
        "velocity_range": {"x": (-0.3, 0.3), "y": (-0.15, 0.15)},
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "throw_box": EventTermCfg(
      func=mdp.throw_box,
      mode="interval",
      interval_range_s=(1.5, 4.0),
      params={
        "box_names": BOX_NAMES,
        "speed_range": (2.0, 5.0),
        "distance_range": (0.6, 1.2),
        "height_range": (0.1, 0.4),
        "asset_cfg": ROBOT_CFG,
      },
    ),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "fallen": TerminationTermCfg(
      func=base_mdp.bad_orientation,
      params={"limit_angle": 0.6, "asset_cfg": ROBOT_CFG},
    ),
    # Reset instead of propagating a diverged physics state into training.
    "nan": TerminationTermCfg(func=base_mdp.nan_detection),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,  # 100 Hz control loop.
    scene=SceneCfg(
      num_envs=16 if play else 4096,
      env_spacing=2.0,
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=uneven_terrain_cfg(),
      ),
      entities={"robot": get_balance_bot_cfg(), **get_box_cfgs()},
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    events=events,
    terminations=terminations,
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.0025)),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=1.5,
      elevation=-15.0,
      azimuth=135.0,
    ),
    episode_length_s=20.0,
  )

  if play:
    cfg.episode_length_s = 1e9
    cfg.events.pop("push_robot", None)

  return cfg


def balance_bot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Variant on flat ground with fixed friction and no thrown boxes."""
  cfg = balance_bot_env_cfg(play=play)
  cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
  cfg.events.pop("ground_friction", None)
  cfg.events.pop("throw_box", None)
  cfg.scene.entities = {"robot": get_balance_bot_cfg()}
  return cfg
