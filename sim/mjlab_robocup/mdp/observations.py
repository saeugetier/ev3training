"""Custom MDP terms for the RoboCup task: sensor readings, actions, rewards.

NOTE ON MJLAB API SURFACE: mjlab's manager-based env exposes named MuJoCo
sensors through helper functions such as ``mdp.builtin_sensor`` (see
``velocity_env_cfg.py``: ``ObservationTermCfg(func=mdp.builtin_sensor,
params={"sensor_name": "robot/imu_ang_vel"})``). The exact low-level
accessor (``_read_named_sensor`` below) wraps that lookup so only one place
needs updating if the installed mjlab version's sensor-access API differs.
Verify against your installed mjlab version before training.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as base_mdp

DEPTH_GRID_SIZE = 8
DEPTH_ZONE_NAMES = tuple(
    f"depth_zone_{r}_{c}" for r in range(DEPTH_GRID_SIZE) for c in range(DEPTH_GRID_SIZE)
)


def _read_named_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Read a single named MuJoCo sensor for all envs, shape [num_envs, dim]."""
    return base_mdp.builtin_sensor(env, sensor_name=sensor_name)


def depth_scan(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Concatenate the 8x8 VL53L8CX zone rangefinders into one [N, 64] tensor.

    Distances are clipped/cutoff at ``DEPTH_MAX_RANGE_M`` in the MJCF sensor
    definition (see assets/robot_asset.py). No-return (max range) readings
    come back as the cutoff value, matching the real sensor's saturation
    behavior closely enough for sim2real purposes.
    """
    readings = [
        _read_named_sensor(env, f"{entity_name}/{name}") for name in DEPTH_ZONE_NAMES
    ]
    return torch.cat(readings, dim=-1)


def yaw_sincos(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Heading as (sin(yaw), cos(yaw)) from the chassis IMU quaternion.

    Using sin/cos instead of the raw angle avoids the +-pi wrap-around
    discontinuity that would otherwise confuse the policy/LSTM state.
    """
    quat = _read_named_sensor(env, f"{entity_name}/imu_quat")  # [N, 4] wxyz
    w, x, y, z = quat.unbind(dim=-1)
    # Yaw from quaternion (z-axis rotation), standard wxyz -> yaw formula.
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def yaw_rate(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Angular rate about the vertical (z) axis from the gyro sensor."""
    ang_vel = _read_named_sensor(env, f"{entity_name}/imu_ang_vel")  # [N, 3] xyz
    return ang_vel[..., 2:3]


def wheel_deltas(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Relative wheel rotation since the last control step (rad).

    Mirrors what the real EV3 `TachoMotor` position feedback provides:
    a delta position read once per control tick, NOT ground-truth wheel
    velocity. Computed here as jointvel * control_dt (decimation * physics
    timestep) to match the discrete sampling the real firmware performs.
    """
    left_vel = _read_named_sensor(env, f"{entity_name}/wheel_left_vel")
    right_vel = _read_named_sensor(env, f"{entity_name}/wheel_right_vel")
    control_dt = env.step_dt
    return torch.cat([left_vel * control_dt, right_vel * control_dt], dim=-1)
