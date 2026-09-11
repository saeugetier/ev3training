//! Dimensions and timing constants shared with the training side.
//!
//! MUST stay in sync with `tools/quantize_export_balance_bot/network_spec.py`
//! and `balance_robot_rl/src/balance_bot/tasks/env_cfg.py` (observation
//! terms, `history_length`, control loop rate).

pub const GYRO_ANGLE_DIM: usize = 1;
pub const GYRO_RATE_DIM: usize = 3;
pub const WHEEL_SPEED_DIM: usize = 2;
pub const WHEEL_DISTANCE_ERROR_DIM: usize = 2;
pub const REMOTE_CONTROL_DIM: usize = 2;
pub const LAST_ACTION_DIM: usize = 2;

/// Per-tick observation width, before history stacking.
pub const STEP_OBS_DIM: usize = GYRO_ANGLE_DIM
    + GYRO_RATE_DIM
    + WHEEL_SPEED_DIM
    + WHEEL_DISTANCE_ERROR_DIM
    + REMOTE_CONTROL_DIM
    + LAST_ACTION_DIM; // 12

pub const HISTORY_LENGTH: usize = 3;
pub const INPUT_DIM: usize = STEP_OBS_DIM * HISTORY_LENGTH; // 36

pub const HIDDEN1_DIM: usize = 128;
pub const HIDDEN2_DIM: usize = 128;
pub const HIDDEN3_DIM: usize = 64;
pub const OUTPUT_DIM: usize = 2; // left_wheel_effort, right_wheel_effort.

/// Control loop period: must match `DECIMATION * PHYSICS_TIMESTEP_S` in
/// `env_cfg.py` (100 Hz).
pub const CONTROL_PERIOD_MS: u64 = 10;
pub const STEP_DT_S: f32 = CONTROL_PERIOD_MS as f32 / 1000.0;

/// Matches `WHEEL_SPEED` in `env_cfg.py`: the odometry target speed used by
/// the remote-control command's integrator, in rad/s per unit control.
pub const WHEEL_SPEED_RAD_S: f32 = 12.0;
/// Matches `RemoteControlCommandCfg.max_pos_error` in `mdp.py` [rad].
pub const MAX_POS_ERROR_RAD: f32 = 2.0;

/// `ObservationTermCfg.scale` factors from `env_cfg.py`, applied to the raw
/// sensor/command value before quantization.
pub const GYRO_RATE_OBS_SCALE: f32 = 0.25;
pub const WHEEL_SPEED_OBS_SCALE: f32 = 0.1;
pub const WHEEL_SPEED_OBS_CLIP_RAD_S: f32 = 100.0;
pub const WHEEL_DISTANCE_ERROR_OBS_SCALE: f32 = 0.5;
