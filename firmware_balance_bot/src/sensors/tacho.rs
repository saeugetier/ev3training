//! Drive-motor wrapper around `ev3dev_lang_rust::motors::LargeMotor`.
//!
//! Effort-controlled (duty-cycle direct drive, no built-in speed
//! regulation), matching the training-side MuJoCo `<motor>` actuator and
//! the EV3's actual lack of a torque or speed servo -- see
//! `balance_robot_rl/README.md` ("Sensors and actuators") for the
//! rationale. `get_position`/`get_speed` are absolute tacho counts and
//! counts/s, which equal degrees and degrees/s for LEGO EV3 motors
//! (`count_per_rot == 360`, verified via `get_count_per_rot` in
//! ev3dev-lang-rust's `tacho_motor.rs`).

use ev3dev_lang_rust::motors::{LargeMotor, MotorPort};
use ev3dev_lang_rust::Ev3Result;

pub struct DriveMotor {
    motor: LargeMotor,
}

impl DriveMotor {
    pub fn new(port: MotorPort) -> Ev3Result<Self> {
        let motor = LargeMotor::get(port)?;
        motor.run_direct()?;
        Ok(Self { motor })
    }

    /// Absolute wheel rotation [rad], matching the sim's `joint_pos`.
    pub fn position_rad(&self) -> Ev3Result<f32> {
        Ok((self.motor.get_position()? as f32).to_radians())
    }

    /// Measured wheel angular velocity [rad/s], matching the sim's `joint_vel`.
    pub fn speed_rad_s(&self) -> Ev3Result<f32> {
        Ok((self.motor.get_speed()? as f32).to_radians())
    }

    /// Command a normalized effort in `[-1, 1]`, mapped linearly to a
    /// `-100..100` duty cycle (the EV3's only motor control primitive).
    pub fn set_effort(&self, effort: f32) -> Ev3Result<()> {
        let duty = (effort.clamp(-1.0, 1.0) * 100.0).round() as i32;
        self.motor.set_duty_cycle_sp(duty)
    }
}
