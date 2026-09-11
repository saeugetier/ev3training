"""Light boxes that get thrown at the robot as a disturbance."""

import mujoco
from mjlab.entity import EntityCfg

BOX_NAMES: tuple[str, ...] = ("box_0", "box_1", "box_2")

_BOX_COLORS: tuple[tuple[float, float, float], ...] = (
  (0.85, 0.25, 0.25),
  (0.25, 0.65, 0.85),
  (0.85, 0.75, 0.25),
)

_BOX_XML = """
<mujoco model="light_box">
  <worldbody>
    <body name="box">
      <freejoint name="root"/>
      <geom name="box" type="box" size="{half} {half} {half}" mass="{mass}"
            rgba="{r} {g} {b} 1" condim="3" friction="0.6 0.02 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""

# Parking spots around the robot; the boxes rest here until they are thrown.
_PARKING_POS: tuple[tuple[float, float], ...] = (
  (1.2, 0.0),
  (-0.6, 1.04),
  (-0.6, -1.04),
)


def get_box_cfg(index: int, half_size: float = 0.04, mass: float = 0.15) -> EntityCfg:
  """A single light box entity (150 g, 8 cm cube by default)."""
  color = _BOX_COLORS[index % len(_BOX_COLORS)]
  xml = _BOX_XML.format(
    half=half_size, mass=mass, r=color[0], g=color[1], b=color[2]
  )
  x, y = _PARKING_POS[index % len(_PARKING_POS)]
  return EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
    init_state=EntityCfg.InitialStateCfg(pos=(x, y, half_size)),
  )


def get_box_cfgs() -> dict[str, EntityCfg]:
  return {name: get_box_cfg(i) for i, name in enumerate(BOX_NAMES)}
