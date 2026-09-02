"""Programmatic MJCF builder for the EV3 RoboCup differential-drive robot.

The VL53L8CX ToF sensor is modeled as an 8x8 grid of MuJoCo `rangefinder`
sensors (one site per zone) instead of a rendered depth camera, since this is
the closest 1:1 match to the real sensor's zone-based ranging output and is
far cheaper to simulate at scale (thousands of parallel envs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Sensor geometry must match the real VL53L8CX mounting/FOV on the physical
# robot. Update these to the as-built values before training a deployable
# policy.
DEPTH_GRID_SIZE = 8  # 8x8 = 64 zones, matches VL53L8CX.
DEPTH_FOV_DEG = 45.0  # VL53L8CX supports up to 45deg x 45deg FOV.
DEPTH_MAX_RANGE_M = 4.0  # VL53L8CX max ranging distance (typical use case).
DEPTH_MOUNT_HEIGHT_M = 0.05  # Height of sensor above chassis center.

WHEEL_RADIUS_M = 0.035
WHEEL_SEPARATION_M = 0.15
WHEEL_AXLE_X_M = 0.03
CASTER_RADIUS_M = 0.018
CASTER_X_M = -0.07
CHASSIS_MASS_KG = 0.3
WHEEL_MASS_KG = 0.02
MAX_WHEEL_SPEED_RAD_S = 2.0 * math.pi * 1.4 # 1 rev/s: stable EV3 training limit.
KICKER_MAX_TORQUE_NM = 0.3


def _depth_sensor_sites_and_sensors() -> tuple[str, str]:
    """Build the <site> and <sensor><rangefinder> blocks for the depth grid."""
    half_fov = math.radians(DEPTH_FOV_DEG) / 2.0
    site_lines: list[str] = []
    sensor_lines: list[str] = []
    n = DEPTH_GRID_SIZE
    for row in range(n):
        for col in range(n):
            # Map grid cell -> (pitch, yaw) offset within the FOV, centered at 0.
            u = (col + 0.5) / n * 2.0 - 1.0  # [-1, 1]
            v = (row + 0.5) / n * 2.0 - 1.0
            yaw = u * half_fov
            pitch = v * half_fov
            name = f"depth_zone_{row}_{col}"
            # A rangefinder measures along its local +z axis. Set that axis
            # directly to avoid Euler-angle composition collapsing the yaw
            # sweep; +x is the robot's forward direction.
            ray_x = math.cos(pitch) * math.cos(yaw)
            ray_y = math.cos(pitch) * math.sin(yaw)
            ray_z = math.sin(pitch)
            site_lines.append(
                f'<site name="{name}" pos="0 0 0" '
                f'zaxis="{ray_x:.4f} {ray_y:.4f} {ray_z:.4f}" '
                f'size="0.002" rgba="1 0 0 0.3"/>'
            )
            sensor_lines.append(
                f'<rangefinder name="{name}" site="{name}" '
                f'cutoff="{DEPTH_MAX_RANGE_M}"/>'
            )
    return "\n        ".join(site_lines), "\n        ".join(sensor_lines)


@dataclass(frozen=True)
class RobotAssetXml:
    xml: str
    depth_sensor_names: tuple[str, ...]


def build_robocup_robot_mjcf() -> RobotAssetXml:
    """Return the full MJCF XML for the differential-drive RoboCup robot."""
    depth_sites, depth_sensors = _depth_sensor_sites_and_sensors()
    depth_names = tuple(
        f"depth_zone_{r}_{c}"
        for r in range(DEPTH_GRID_SIZE)
        for c in range(DEPTH_GRID_SIZE)
    )

    xml = f"""
<mujoco model="ev3_robocup_robot">
  <compiler angle="radian" autolimits="true"/>

  <default>
    <joint damping="0.01" armature="0.001"/>
    <geom friction="0.9 0.01 0.01" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  </default>

  <worldbody>
    <body name="chassis" pos="0 0 {WHEEL_RADIUS_M}">
      <freejoint name="chassis_free"/>
      <inertial pos="0 0 -0.01" mass="{CHASSIS_MASS_KG}" diaginertia="4e-4 4e-4 6e-4"/>
      <geom name="chassis_geom" type="box" size="0.08 0.06 0.03" rgba="0.2 0.2 0.2 1"/>

      <!-- Depth sensor mount (VL53L8CX), front-facing. -->
      <body name="depth_sensor_mount" pos="0.08 0 {DEPTH_MOUNT_HEIGHT_M}">
        {depth_sites}
      </body>

      <!-- IMU / gyro site at chassis center. -->
      <site name="imu_site" pos="0 0 0" size="0.001"/>

      <!-- Kicker arm, single hinge joint, spring-return. -->
      <body name="kicker_arm" pos="0.09 0 -0.01">
        <joint name="kicker_joint" type="hinge" axis="0 1 0"
               range="0 1.2" damping="0.05" springref="0" stiffness="0.2"/>
        <inertial pos="0.02 0 0" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
        <!-- contype=2/conaffinity=0: the kicker must only ever contact the
             ball (see build_ball_mjcf's conaffinity), never the ground/walls
             -- otherwise the arm drags on the floor and brakes the chassis
             whenever a kick is held active. -->
        <geom name="kicker_geom" type="box" size="0.03 0.015 0.005"
              contype="2" conaffinity="0" rgba="0.8 0.1 0.1 1"/>
      </body>

      <!-- Larger drive wheels, with the axle forward of the chassis center. -->
      <body name="wheel_left" pos="{WHEEL_AXLE_X_M} {WHEEL_SEPARATION_M / 2.0} 0"
            euler="1.5708 0 0">
        <joint name="wheel_left_joint" type="hinge" axis="0 0 1" damping="0.02"/>
        <inertial pos="0 0 0" mass="{WHEEL_MASS_KG}" diaginertia="1e-5 1e-5 2e-5"/>
        <geom name="wheel_left_geom" type="cylinder"
              size="{WHEEL_RADIUS_M} 0.008" rgba="0.1 0.1 0.1 1"/>
      </body>

      <!-- Right wheel. -->
      <body name="wheel_right" pos="{WHEEL_AXLE_X_M} {-WHEEL_SEPARATION_M / 2.0} 0"
            euler="1.5708 0 0">
        <joint name="wheel_right_joint" type="hinge" axis="0 0 1" damping="0.02"/>
        <inertial pos="0 0 0" mass="{WHEEL_MASS_KG}" diaginertia="1e-5 1e-5 2e-5"/>
        <geom name="wheel_right_geom" type="cylinder"
              size="{WHEEL_RADIUS_M} 0.008" rgba="0.1 0.1 0.1 1"/>
      </body>

      <!-- Free-spinning rear ball caster keeps the chassis level on three wheels. -->
      <body name="rear_caster" pos="{CASTER_X_M} 0 {-WHEEL_RADIUS_M + CASTER_RADIUS_M}">
        <joint name="rear_caster_joint" type="ball" damping="0.01"/>
        <geom name="rear_caster_geom" type="sphere" size="{CASTER_RADIUS_M}"
              mass="0.01" rgba="0.35 0.35 0.35 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="wheel_left_act" joint="wheel_left_joint"
              ctrlrange="-{MAX_WHEEL_SPEED_RAD_S} {MAX_WHEEL_SPEED_RAD_S}" kv="0.6"/>
    <velocity name="wheel_right_act" joint="wheel_right_joint"
              ctrlrange="-{MAX_WHEEL_SPEED_RAD_S} {MAX_WHEEL_SPEED_RAD_S}" kv="0.6"/>
    <motor name="kicker_act" joint="kicker_joint"
           ctrlrange="0 {KICKER_MAX_TORQUE_NM}"/>
  </actuator>

  <sensor>
    <gyro name="imu_ang_vel" site="imu_site"/>
    <framequat name="imu_quat" objtype="site" objname="imu_site"/>
    <jointvel name="wheel_left_vel" joint="wheel_left_joint"/>
    <jointvel name="wheel_right_vel" joint="wheel_right_joint"/>
    <jointpos name="wheel_left_pos" joint="wheel_left_joint"/>
    <jointpos name="wheel_right_pos" joint="wheel_right_joint"/>
    <jointpos name="kicker_pos" joint="kicker_joint"/>
    {depth_sensors}
  </sensor>
</mujoco>
"""
    return RobotAssetXml(xml=xml, depth_sensor_names=depth_names)
