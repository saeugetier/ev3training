//! Apply the policy's raw (unbounded mean, no squashing activation) action
//! as a normalized wheel effort, matching the training-side MuJoCo
//! `<motor>` actuator's `ctrlrange="-1 1"` clamp and the EV3's
//! duty-cycle-only motor control -- see `balance_robot_rl/README.md`.

use ev3dev_lang_rust::Ev3Result;

use crate::config::MAX_TEST_DUTY;
use crate::policy_weights::ACTION_SCALE;
use crate::sensors::tacho::DriveMotor;

/// Dequantize a Q15 action logit to a real effort value, clamped to the
/// `[-1, 1]` range the actuator (sim and hardware) actually accepts.
pub fn dequantize_effort(raw: i16) -> f32 {
    (raw as f32 * ACTION_SCALE).clamp(-1.0, 1.0)
}

pub fn apply_action(
    action: &[i16; 2],
    left: &DriveMotor,
    right: &DriveMotor,
) -> Ev3Result<[f32; 2]> {
    let left_effort = dequantize_effort(action[0]).clamp(-MAX_TEST_DUTY, MAX_TEST_DUTY);
    let right_effort = dequantize_effort(action[1]).clamp(-MAX_TEST_DUTY, MAX_TEST_DUTY);

    left.set_effort(left_effort)?;
    right.set_effort(right_effort)?;

    Ok([left_effort, right_effort])
}
