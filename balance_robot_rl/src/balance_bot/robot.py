"""Entity configuration for the two-wheel balance bot."""

from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

BALANCE_BOT_XML: Path = Path(__file__).parent / "assets" / "balance_bot.xml"

WHEEL_NAMES: tuple[str, str] = ("left_wheel", "right_wheel")
WHEEL_RADIUS: float = 0.05
WHEEL_SEPARATION: float = 0.15
"""Distance between the two wheel contact points [m]."""


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(BALANCE_BOT_XML))


def get_balance_bot_cfg() -> EntityCfg:
  """Balance bot entity: free-floating chassis with two torque-driven wheels."""
  return EntityCfg(
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=WHEEL_NAMES),),
    ),
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.05),
      joint_pos={".*_wheel": 0.0},
      joint_vel={".*_wheel": 0.0},
    ),
  )
