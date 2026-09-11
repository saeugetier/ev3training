//! Build the Q15-quantized, history-stacked observation vector matching
//! the exact term order and `history_length=3` used in sim (see
//! `balance_bot_env_cfg`'s `sensor_terms` in
//! `balance_robot_rl/src/balance_bot/tasks/env_cfg.py`):
//! `[gyro_angle(1), gyro_rate(3), wheel_speed(2), wheel_distance_error(2),
//!  remote_control(2), last_action(2)]` = 12 values/tick, each stacked over
//! the last 3 ticks.
//!
//! ASSUMPTION / TODO: mjlab's `ObservationManager` is assumed to stack
//! history *per term* (all `history_length` samples of one term
//! contiguous, oldest-to-newest, before moving to the next term) rather
//! than *per tick* (one full tick's terms, repeated `history_length`
//! times). Verify against the installed mjlab version -- see
//! `tools/quantize_export_balance_bot/network_spec.py`.

use crate::config::{
    GYRO_ANGLE_DIM, GYRO_RATE_DIM, GYRO_RATE_OBS_SCALE, HISTORY_LENGTH, INPUT_DIM,
    LAST_ACTION_DIM, REMOTE_CONTROL_DIM, STEP_OBS_DIM, WHEEL_DISTANCE_ERROR_DIM,
    WHEEL_DISTANCE_ERROR_OBS_SCALE, WHEEL_SPEED_DIM, WHEEL_SPEED_OBS_CLIP_RAD_S,
    WHEEL_SPEED_OBS_SCALE,
};
use crate::policy_weights::OBS_SCALE;

/// Term boundaries (offset, width) within one tick's `STEP_OBS_DIM` vector.
const TERM_BOUNDS: [(usize, usize); 6] = [
    (0, GYRO_ANGLE_DIM),
    (GYRO_ANGLE_DIM, GYRO_RATE_DIM),
    (GYRO_ANGLE_DIM + GYRO_RATE_DIM, WHEEL_SPEED_DIM),
    (
        GYRO_ANGLE_DIM + GYRO_RATE_DIM + WHEEL_SPEED_DIM,
        WHEEL_DISTANCE_ERROR_DIM,
    ),
    (
        GYRO_ANGLE_DIM + GYRO_RATE_DIM + WHEEL_SPEED_DIM + WHEEL_DISTANCE_ERROR_DIM,
        REMOTE_CONTROL_DIM,
    ),
    (
        GYRO_ANGLE_DIM
            + GYRO_RATE_DIM
            + WHEEL_SPEED_DIM
            + WHEEL_DISTANCE_ERROR_DIM
            + REMOTE_CONTROL_DIM,
        LAST_ACTION_DIM,
    ),
];

/// Ring buffer of the last `HISTORY_LENGTH` per-tick observation vectors.
pub struct ObservationHistory {
    buf: [[f32; STEP_OBS_DIM]; HISTORY_LENGTH],
    next: usize,
}

impl ObservationHistory {
    pub fn new() -> Self {
        Self {
            buf: [[0.0; STEP_OBS_DIM]; HISTORY_LENGTH],
            next: 0,
        }
    }

    pub fn push(&mut self, step_obs: [f32; STEP_OBS_DIM]) {
        self.buf[self.next] = step_obs;
        self.next = (self.next + 1) % HISTORY_LENGTH;
    }

    /// Term-major history stack (see module docs for the stacking
    /// assumption), as raw (unquantized) floats.
    pub fn flatten(&self) -> [f32; INPUT_DIM] {
        let mut out = [0.0f32; INPUT_DIM];
        // Oldest-to-newest chronological order of the ring buffer.
        let order: [usize; HISTORY_LENGTH] = core::array::from_fn(|k| (self.next + k) % HISTORY_LENGTH);

        let mut out_idx = 0;
        for (offset, width) in TERM_BOUNDS {
            for &step in order.iter() {
                out[out_idx..out_idx + width].copy_from_slice(&self.buf[step][offset..offset + width]);
                out_idx += width;
            }
        }
        out
    }
}

impl Default for ObservationHistory {
    fn default() -> Self {
        Self::new()
    }
}

fn quantize(real_value: f32) -> i16 {
    (real_value / OBS_SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Build one tick's raw (unquantized) observation vector, applying the same
/// `ObservationTermCfg.scale`/`clip` factors as `env_cfg.py`.
#[allow(clippy::too_many_arguments)]
pub fn build_step_obs(
    gyro_angle_rad: f32,
    gyro_tilt_rate_rad_s: f32,
    wheel_speed_rad_s: [f32; 2],
    wheel_distance_error_rad: [f32; 2],
    remote_controls: [f32; 2],
    last_action: [f32; 2],
) -> [f32; STEP_OBS_DIM] {
    let mut step = [0.0f32; STEP_OBS_DIM];
    let mut idx = 0;

    step[idx] = gyro_angle_rad;
    idx += 1;

    // Only the wheel-axis (tilt) rate is physically measured by the EV3's
    // single-axis gyro; the other two body-frame axes of the sim's 3-axis
    // `base_ang_vel` term have no on-device sensor and are approximated as
    // zero (roll is structurally near-zero for this robot; yaw could in
    // principle be estimated from the wheel speed difference, but that is
    // not implemented here).
    step[idx] = 0.0;
    step[idx + 1] = gyro_tilt_rate_rad_s * GYRO_RATE_OBS_SCALE;
    step[idx + 2] = 0.0;
    idx += GYRO_RATE_DIM;

    for v in wheel_speed_rad_s {
        step[idx] = v.clamp(-WHEEL_SPEED_OBS_CLIP_RAD_S, WHEEL_SPEED_OBS_CLIP_RAD_S) * WHEEL_SPEED_OBS_SCALE;
        idx += 1;
    }

    for v in wheel_distance_error_rad {
        step[idx] = v * WHEEL_DISTANCE_ERROR_OBS_SCALE;
        idx += 1;
    }

    for v in remote_controls {
        step[idx] = v;
        idx += 1;
    }

    for v in last_action {
        step[idx] = v;
        idx += 1;
    }

    debug_assert_eq!(idx, STEP_OBS_DIM);
    step
}

/// Quantize a flattened, history-stacked observation to Q15.
pub fn quantize_obs(flat: &[f32; INPUT_DIM]) -> [i16; INPUT_DIM] {
    let mut out = [0i16; INPUT_DIM];
    for (o, &v) in out.iter_mut().zip(flat.iter()) {
        *o = quantize(v);
    }
    out
}
