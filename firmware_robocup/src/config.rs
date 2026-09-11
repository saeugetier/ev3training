//! Dimensions and timing constants shared with the training side.
//!
//! MUST stay in sync with `tools/quantize_export/network_spec.py` and
//! `sim/mjlab_robocup/robocup_env_cfg.py` (control loop rate).

pub const DEPTH_DIM: usize = 64;
pub const YAW_SINCOS_DIM: usize = 2;
pub const YAW_RATE_DIM: usize = 1;
pub const WHEEL_DELTAS_DIM: usize = 2;
pub const LAST_ACTION_DIM: usize = 3;

pub const INPUT_DIM: usize =
    DEPTH_DIM + YAW_SINCOS_DIM + YAW_RATE_DIM + WHEEL_DELTAS_DIM + LAST_ACTION_DIM;
pub const LSTM_HIDDEN_DIM: usize = 96;
pub const DECODER_DIM: usize = 96;
pub const OUTPUT_DIM: usize = 3;

/// Control loop period: must match `DECIMATION * PHYSICS_TIMESTEP_S` in
/// `robocup_env_cfg.py` (4 * 5ms = 20ms -> 50 Hz).
pub const CONTROL_PERIOD_MS: u64 = 20;

/// Real robot geometry, must match `assets/robot_asset.py`.
pub const MAX_WHEEL_SPEED_DEG_S: i32 = 360; // 1 rev/s, matches simulation.
pub const KICK_TRIGGER_THRESHOLD_Q15: i16 = 16384; // 0.5 in Q15.

/// VL53L8CX depth sensor.
pub const DEPTH_MAX_RANGE_MM: u16 = 4000;
