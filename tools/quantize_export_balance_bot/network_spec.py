"""Single source of truth for the deployed balance-bot policy network topology.

Both the Python export pipeline and any future Rust firmware must agree on
these dimensions. If you change `hidden_dims` in
`balance_robot_rl/src/balance_bot/tasks/rl_cfg.py` or the observation terms
/ `history_length` in `balance_robot_rl/src/balance_bot/tasks/env_cfg.py`,
update this file together with them.

Topology: a plain feed-forward MLP (matching rsl_rl's stock `ActorCritic`
actor -- no recurrent state, unlike the RoboCup kick policy):

    obs -> Linear(INPUT_DIM -> 128) -> LeakyReLU
        -> Linear(128 -> 128)       -> LeakyReLU
        -> Linear(128 -> 64)        -> LeakyReLU
        -> Linear(64 -> OUTPUT_DIM)              [raw mean action, no
                                                   squashing activation --
                                                   `GaussianDistribution` in
                                                   rl_cfg.py has no tanh head]

ASSUMPTION / TODO: the per-step observation term order below is assumed to
match `balance_bot_env_cfg`'s `sensor_terms` dict insertion order
(`gyro_angle, gyro_rate, wheel_speed, wheel_distance_error, remote_control,
last_action`), stacked over `history_length=3` samples. Verify this against
the installed mjlab version's `ObservationManager` stacking convention
(term-major vs. time-major) before trusting the exported input layout.
"""

from __future__ import annotations

GYRO_ANGLE_DIM = 1
GYRO_RATE_DIM = 3
WHEEL_SPEED_DIM = 2
WHEEL_DISTANCE_ERROR_DIM = 2
REMOTE_CONTROL_DIM = 2
LAST_ACTION_DIM = 2

STEP_OBS_DIM = (
  GYRO_ANGLE_DIM
  + GYRO_RATE_DIM
  + WHEEL_SPEED_DIM
  + WHEEL_DISTANCE_ERROR_DIM
  + REMOTE_CONTROL_DIM
  + LAST_ACTION_DIM
)  # 12
HISTORY_LENGTH = 3
INPUT_DIM = STEP_OBS_DIM * HISTORY_LENGTH  # 36

HIDDEN_DIMS = (128, 128, 64)
OUTPUT_DIM = 2  # left_wheel_effort, right_wheel_effort, each in [-1, 1].

assert INPUT_DIM == 36
