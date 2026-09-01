"""Custom action term: 3 continuous actions -> 2 wheel velocities + kick torque.

Writes directly to the entity's actuator `ctrl` buffer (via
`EntityData.write_ctrl`) rather than composing multiple built-in
``JointVelocityActionCfg``/``JointEffortActionCfg`` terms, since the kick
channel needs the discrete threshold-fire behavior mirrored in
`firmware/src/actuation.rs`, not a continuous torque passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

WHEEL_LEFT_ACTUATOR = "wheel_left_act"
WHEEL_RIGHT_ACTUATOR = "wheel_right_act"
KICKER_ACTUATOR = "kicker_act"

# Action[2] (kick) fires the kicker when it crosses this threshold. A
# threshold rather than a raw torque pass-through keeps the kick a
# discrete, easily-mirrored-on-hardware event instead of continuous force.
KICK_TRIGGER_THRESHOLD = 0.5


@dataclass(kw_only=True)
class RoboCupDriveActionCfg(ActionTermCfg):
    max_wheel_speed_rad_s: float = 6.283  # Must match assets/robot_asset.py.
    max_kick_torque_nm: float = 0.3
    log_actions: bool = False
    log_action_interval: int = 10

    def build(self, env: ManagerBasedRlEnv) -> RoboCupDriveAction:
        return RoboCupDriveAction(self, env)


class RoboCupDriveAction(ActionTerm):
    """Maps a 3-dim tanh-bounded action to [wheel_l, wheel_r, kick]."""

    cfg: RoboCupDriveActionCfg

    def __init__(self, cfg: RoboCupDriveActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        joint_ids, _ = self._entity.find_joints(
            ("wheel_left_joint", "wheel_right_joint", "kicker_joint"),
            preserve_order=True,
        )
        self._wheel_joint_ids = torch.tensor(
            joint_ids[:2], device=self.device, dtype=torch.long
        )
        self._kicker_joint_id = torch.tensor(
            joint_ids[2:], device=self.device, dtype=torch.long
        )
        self._raw_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device
        )
        self._action_steps = 0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        # Policy output layer is assumed tanh-bounded to [-1, 1] (see
        # tools/quantize_export network spec) so no extra clipping needed,
        # but clip defensively in case of exploration noise overshoot.
        self._raw_actions = torch.clamp(actions, -1.0, 1.0)
        self._action_steps += 1
        if self.cfg.log_actions and self._action_steps % self.cfg.log_action_interval == 0:
            left, right, kick = self._raw_actions[0].tolist()
            print(
                f"[policy action] left={left:+.3f} right={right:+.3f} "
                f"kick={kick:+.3f}"
            )

    def apply_actions(self) -> None:
        # Both wheel-joint axes point along -y in the MJCF, so negating the
        # policy output makes equal positive actions drive forward along +x,
        # which is also the depth sensor's viewing direction.
        left_cmd = -self._raw_actions[:, 0] * self.cfg.max_wheel_speed_rad_s
        right_cmd = -self._raw_actions[:, 1] * self.cfg.max_wheel_speed_rad_s
        kick_active = (self._raw_actions[:, 2] > KICK_TRIGGER_THRESHOLD).float()
        kick_cmd = kick_active * self.cfg.max_kick_torque_nm

        self._entity.set_joint_velocity_target(
            torch.stack([left_cmd, right_cmd], dim=-1),
            joint_ids=self._wheel_joint_ids,
        )
        self._entity.set_joint_effort_target(
            kick_cmd.unsqueeze(-1), joint_ids=self._kicker_joint_id
        )

