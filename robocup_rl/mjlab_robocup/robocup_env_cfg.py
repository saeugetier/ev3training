"""RoboCup dribble-and-kick task configuration (mjlab ManagerBasedRlEnvCfg).

ASSUMPTION / TODO: mjlab's exact `EntityCfg`/`SceneCfg` field names for
composing multiple raw-MJCF bodies (robot + ball + field) into one scene
are not pinned here -- verify against your installed mjlab version (see
the "custom robot integration" example: mujocolab/anymal_c_velocity) and
adjust `SceneCfg(...)` construction below if field names differ.
"""

from __future__ import annotations

import math
import os

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_robocup import mdp
from mjlab_robocup.assets.field_asset import build_ball_mjcf, build_field_mjcf
from mjlab_robocup.assets.robot_asset import build_robocup_robot_mjcf

# Control loop rate: physics timestep * decimation must match the real EV3
# firmware's control tick (see firmware/src/main.rs CONTROL_PERIOD_MS).
PHYSICS_TIMESTEP_S = 0.005
DECIMATION = 4  # -> 50 Hz control loop, matches firmware target.


def make_robocup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    robot_asset = build_robocup_robot_mjcf()

    scene = SceneCfg(
        entities={
            "robot": EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_string(robot_asset.xml),
                # Actuators are already fully specified in the MJCF; this just
                # marks the entity as actuated so mjlab tracks a `ctrl_ids`
                # mapping for it (write_ctrl requires `is_actuated`).
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        XmlActuatorCfg(target_names_expr=("wheel_left_joint",)),
                        XmlActuatorCfg(target_names_expr=("wheel_right_joint",)),
                        XmlActuatorCfg(target_names_expr=("kicker_joint",)),
                    )
                ),
            ),
            "ball": EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_string(build_ball_mjcf())
            ),
            "field": EntityCfg(
                spec_fn=lambda: mujoco.MjSpec.from_string(build_field_mjcf())
            ),
        },
        num_envs=1,
        extent=3.0,
    )

    # Observations: strictly what the real EV3 firmware can measure
    # (depth grid, gyro yaw/rate, wheel encoder deltas, last action).
    # No privileged ball/goal position in the deployed "actor" group.
    actor_terms = {
        "depth_scan": ObservationTermCfg(
            func=mdp.depth_scan,
            noise=Unoise(n_min=-0.02, n_max=0.02),  # ~2cm VL53L8CX noise.
        ),
        "yaw_sincos": ObservationTermCfg(
            func=mdp.yaw_sincos, noise=Unoise(n_min=-0.02, n_max=0.02)
        ),
        "yaw_rate": ObservationTermCfg(
            func=mdp.yaw_rate, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "wheel_deltas": ObservationTermCfg(
            func=mdp.wheel_deltas, noise=Unoise(n_min=-0.01, n_max=0.01)
        ),
        "last_action": ObservationTermCfg(func=base_mdp.last_action),
    }
    # Critic gets privileged ball/goal-relative info to speed up training;
    # never deployed to hardware.
    critic_terms = {
        **actor_terms,
        "robot_ball_vec": ObservationTermCfg(func=mdp.robot_ball_vec),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=not play,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms, concatenate_terms=True, enable_corruption=False
        ),
    }

    actions = {
        "drive": mdp.RoboCupDriveActionCfg(
            entity_name="robot",
            log_actions=os.getenv("MJLAB_ROBOCUP_LOG_ACTIONS") == "1",
            log_action_interval=int(
                os.getenv("MJLAB_ROBOCUP_LOG_ACTION_INTERVAL", "10")
            ),
        ),
    }

    rewards = {
        "robot_ball_approach": RewardTermCfg(
            func=mdp.robot_ball_approach, weight=4.0
        ),
        "ball_goal_progress": RewardTermCfg(func=mdp.ball_goal_progress, weight=8.0),
        "goal_scored": RewardTermCfg(func=mdp.goal_scored_bonus, weight=50.0),
        "ball_possession": RewardTermCfg(func=mdp.ball_possession, weight=0.002),
        "dribble_goal_alignment": RewardTermCfg(
            func=mdp.dribble_goal_alignment, weight=0.004
        ),
        "kick_misuse": RewardTermCfg(func=mdp.kick_misuse_penalty, weight=-2.0),
        "kick_on_target": RewardTermCfg(func=mdp.kick_on_target_bonus, weight=5.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "wheel_energy_l2": RewardTermCfg(func=mdp.wheel_energy_l2, weight=-0.01),
        "robot_wall_proximity": RewardTermCfg(
            func=mdp.robot_wall_proximity, weight=-1.0
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "goal_scored": TerminationTermCfg(func=mdp.goal_scored, time_out=True),
        "out_of_bounds": TerminationTermCfg(func=mdp.out_of_field_bounds),
        "nan_detection": TerminationTermCfg(func=base_mdp.nan_detection),
    }
    events = {
        "reset_ball": EventTermCfg(func=mdp.reset_ball_randomly, mode="reset"),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=PHYSICS_TIMESTEP_S,
                iterations=10,
                ls_iterations=10,
            ),
            njmax=128,
        ),
        decimation=DECIMATION,
        episode_length_s=60.0,
    )
