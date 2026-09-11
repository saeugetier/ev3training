//! Decode the policy's Q0.15 tanh-bounded action into real motor commands.
//!
//! Mirrors `mjlab_robocup.mdp.actions.RoboCupDriveAction` (sim side): a
//! threshold on the 3rd action component fires the kicker instead of a
//! continuous torque, so the same discrete kick event is easy to compare
//! between sim and hardware.

use ev3dev_lang_rust::Ev3Result;

use crate::config::{KICK_TRIGGER_THRESHOLD_Q15, MAX_WHEEL_SPEED_DEG_S};
use crate::policy_weights::UNIT_SCALE;
use crate::sensors::tacho::{DriveMotor, KickerMotor};

pub fn apply_action(
    action: &[i16; 3],
    left: &DriveMotor,
    right: &DriveMotor,
    kicker: &KickerMotor,
) -> Ev3Result<()> {
    // Positive policy actions mean forward, matching the MuJoCo wheel-axis convention.
    let left_cmd_deg_s = -(action[0] as f32 * UNIT_SCALE * MAX_WHEEL_SPEED_DEG_S as f32) as i32;
    let right_cmd_deg_s = -(action[1] as f32 * UNIT_SCALE * MAX_WHEEL_SPEED_DEG_S as f32) as i32;

    left.set_speed_deg_s(left_cmd_deg_s, MAX_WHEEL_SPEED_DEG_S)?;
    right.set_speed_deg_s(right_cmd_deg_s, MAX_WHEEL_SPEED_DEG_S)?;

    if action[2] > KICK_TRIGGER_THRESHOLD_Q15 {
        kicker.fire_kick(100)?;
    } else {
        kicker.stop()?;
    }
    Ok(())
}
