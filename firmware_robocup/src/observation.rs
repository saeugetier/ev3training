//! Build the Q15-quantized observation vector matching the exact feature
//! order used in sim (see sim/mjlab_robocup/robocup_env_cfg.py):
//! [depth(64), yaw_sin, yaw_cos, yaw_rate, wheel_left_delta,
//!  wheel_right_delta, last_action(3)] = 72 values, all quantized at
//! `policy_weights::OBS_SCALE`.

use crate::config::{DEPTH_DIM, DEPTH_MAX_RANGE_MM, INPUT_DIM};
use crate::policy_weights::{OBS_SCALE, UNIT_SCALE};
use crate::sensors::vl53l8cx::DepthFrame;

fn quantize(real_value: f32) -> i16 {
    (real_value / OBS_SCALE)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[allow(clippy::too_many_arguments)]
pub fn build_observation(
    depth: &DepthFrame,
    yaw_deg: i32,
    yaw_rate_deg_s: i32,
    wheel_left_delta_deg: i32,
    wheel_right_delta_deg: i32,
    last_action: &[i16; 3],
) -> [i16; INPUT_DIM] {
    let mut obs = [0i16; INPUT_DIM];
    let mut idx = 0;

    for i in 0..DEPTH_DIM {
        // Real distance in meters, matching the simulated rangefinder's
        // unit (assets/robot_asset.py uses meters, cutoff at DEPTH_MAX_RANGE_MM).
        let mm = depth.distances_mm[i].min(DEPTH_MAX_RANGE_MM) as f32;
        obs[idx] = quantize(mm / 1000.0);
        idx += 1;
    }

    let yaw_rad = (yaw_deg as f32).to_radians();
    // ev3dev runs full Linux (std available); f32::sin/cos work fine on the
    // soft-float ABI, just software-emulated (slower, but called only twice
    // per control tick here -- acceptable, unlike the per-weight NN math
    // which must stay pure fixed-point for real-time performance).
    obs[idx] = quantize(yaw_rad.sin());
    idx += 1;
    obs[idx] = quantize(yaw_rad.cos());
    idx += 1;

    obs[idx] = quantize((yaw_rate_deg_s as f32).to_radians());
    idx += 1;

    // wheel_deltas in sim are radians (jointvel * control_dt); convert
    // degrees (ev3dev tacho position units) to radians here.
    obs[idx] = quantize((wheel_left_delta_deg as f32).to_radians());
    idx += 1;
    obs[idx] = quantize((wheel_right_delta_deg as f32).to_radians());
    idx += 1;

    for k in 0..3 {
        // `last_action` is the raw tanh_s16 output from the previous tick
        // (Q0.15, see policy.rs), i.e. already scaled by UNIT_SCALE.
        obs[idx] = quantize(last_action[k] as f32 * UNIT_SCALE);
        idx += 1;
    }

    debug_assert_eq!(idx, INPUT_DIM);
    obs
}
