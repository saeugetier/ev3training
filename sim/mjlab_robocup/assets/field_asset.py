"""Static MJCF for the ball and a bounded RoboCup field with goal markers.

Field/goal dimensions match the constants in mdp/rewards.py -- keep them
in sync if you change either file.
"""

from __future__ import annotations

BALL_RADIUS_M = 0.051  # Standard size-1 mini ball scaled for tabletop field.
BALL_MASS_KG = 0.045

FIELD_HALF_LENGTH_M = 2.2
FIELD_HALF_WIDTH_M = 1.5
GOAL_HALF_WIDTH_M = 0.35
GOAL_POS_X = 2.0
WALL_THICKNESS_M = 0.03
WALL_HEIGHT_M = 0.30
ROBOT_WALL_CLEARANCE_M = 0.20


def build_ball_mjcf() -> str:
    return f"""
<mujoco model="robocup_ball">
  <worldbody>
    <body name="ball" pos="0 0 {BALL_RADIUS_M}">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{BALL_MASS_KG}"
                diaginertia="1e-5 1e-5 1e-5"/>
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS_M}"
            friction="0.4 0.01 0.01" rgba="1 0.5 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_field_mjcf() -> str:
    return f"""
<mujoco model="robocup_field">
  <worldbody>
    <geom name="ground" type="plane"
          size="{FIELD_HALF_LENGTH_M} {FIELD_HALF_WIDTH_M} 0.01"
          rgba="0.05 0.4 0.05 1" friction="0.9 0.01 0.01"/>
        <geom name="wall_positive_x" type="box"
          pos="{FIELD_HALF_LENGTH_M} 0 {WALL_HEIGHT_M / 2.0}"
          size="{WALL_THICKNESS_M / 2.0} {FIELD_HALF_WIDTH_M} {WALL_HEIGHT_M / 2.0}"
          rgba="0.85 0.85 0.85 1" friction="0.9 0.01 0.01"/>
        <geom name="wall_negative_x" type="box"
          pos="{-FIELD_HALF_LENGTH_M} 0 {WALL_HEIGHT_M / 2.0}"
          size="{WALL_THICKNESS_M / 2.0} {FIELD_HALF_WIDTH_M} {WALL_HEIGHT_M / 2.0}"
          rgba="0.85 0.85 0.85 1" friction="0.9 0.01 0.01"/>
        <geom name="wall_positive_y" type="box"
          pos="0 {FIELD_HALF_WIDTH_M} {WALL_HEIGHT_M / 2.0}"
          size="{FIELD_HALF_LENGTH_M} {WALL_THICKNESS_M / 2.0} {WALL_HEIGHT_M / 2.0}"
          rgba="0.85 0.85 0.85 1" friction="0.9 0.01 0.01"/>
        <geom name="wall_negative_y" type="box"
          pos="0 {-FIELD_HALF_WIDTH_M} {WALL_HEIGHT_M / 2.0}"
          size="{FIELD_HALF_LENGTH_M} {WALL_THICKNESS_M / 2.0} {WALL_HEIGHT_M / 2.0}"
          rgba="0.85 0.85 0.85 1" friction="0.9 0.01 0.01"/>
    <geom name="goal_post_top" type="box"
          pos="{GOAL_POS_X} {GOAL_HALF_WIDTH_M} 0.05"
          size="0.03 0.03 0.1" rgba="1 1 1 1"/>
    <geom name="goal_post_bottom" type="box"
          pos="{GOAL_POS_X} {-GOAL_HALF_WIDTH_M} 0.05"
          size="0.03 0.03 0.1" rgba="1 1 1 1"/>
  </worldbody>
</mujoco>
"""
