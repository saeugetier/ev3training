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

use crate::config::{LEFT_MOTOR_SIGN, RIGHT_MOTOR_SIGN, LEFT_MOTOR_ACTOR_SCALE, RIGHT_MOTOR_ACTOR_SCALE, LEFT_MOTOR_OBS_SCALE, RIGHT_MOTOR_OBS_SCALE, MAX_TEST_DUTY};

pub struct DriveMotor {
    motor: LargeMotor,
    sign: f32,
    act_scale: f32,
    obs_scale: f32,
}

impl DriveMotor {
    pub fn new(port: MotorPort) -> Ev3Result<Self> {
        let motor = LargeMotor::get(port)?;
        motor.set_position(0)?;
        motor.run_direct()?;
        let sign = match port {
            MotorPort::OutA => LEFT_MOTOR_SIGN,
            MotorPort::OutD => RIGHT_MOTOR_SIGN,
            _ => 1.0,
        };
        let act_scale = match port {
            MotorPort::OutA => LEFT_MOTOR_ACTOR_SCALE,
            MotorPort::OutD => RIGHT_MOTOR_ACTOR_SCALE,
            _ => 1.0,
        };
        let obs_scale = match port {
            MotorPort::OutA => LEFT_MOTOR_OBS_SCALE,
            MotorPort::OutD => RIGHT_MOTOR_OBS_SCALE,
            _ => 1.0,
        };
        Ok(Self { motor, sign, act_scale, obs_scale })
    }

    /// Absolute wheel rotation [rad], matching the sim's `joint_pos`.
    pub fn position_rad(&self) -> Ev3Result<f32> {
        Ok(self.sign *  self.obs_scale * (self.motor.get_position()? as f32).to_radians())
    }

    /// Measured wheel angular velocity [rad/s], matching the sim's `joint_vel`.
    pub fn speed_rad_s(&self) -> Ev3Result<f32> {
        Ok(self.sign * self.obs_scale * (self.motor.get_speed()? as f32).to_radians())
    }

    /// Command a normalized effort in `[-1, 1]`, mapped linearly to a
    /// `-100..100` duty cycle (the EV3's only motor control primitive).
    pub fn set_effort(&self, effort: f32) -> Ev3Result<()> {
        let effort = (effort * self.act_scale).clamp(-MAX_TEST_DUTY, MAX_TEST_DUTY);
        let duty = (self.sign * effort * 100.0).round() as i32;
        self.motor.set_duty_cycle_sp(duty)
    }

    pub fn stop(&self) -> Ev3Result<()> {
        self.motor.set_duty_cycle_sp(0)
    }
}
