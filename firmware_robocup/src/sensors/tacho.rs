//! Drive-motor and kicker-motor wrappers around `ev3dev_lang_rust::motors`.
//!
//! `LargeMotor::get_position()` gives absolute encoder counts (degrees);
//! the firmware tracks the delta since the previous control tick to
//! mirror the sim's `wheel_deltas` observation (relative rotation, not
//! ground-truth velocity -- see sim/mjlab_robocup/mdp/observations.py).

use ev3dev_lang_rust::motors::{LargeMotor, MotorPort};
use ev3dev_lang_rust::Ev3Result;

pub struct DriveMotor {
    motor: LargeMotor,
    last_position_deg: i32,
}

impl DriveMotor {
    pub fn new(port: MotorPort) -> Ev3Result<Self> {
        let motor = LargeMotor::get(port)?;
        motor.run_direct()?;
        let last_position_deg = motor.get_position()?;
        Ok(Self {
            motor,
            last_position_deg,
        })
    }

    /// Relative rotation (degrees) since the last call to this function.
    pub fn poll_delta_deg(&mut self) -> Ev3Result<i32> {
        let pos = self.motor.get_position()?;
        let delta = pos - self.last_position_deg;
        self.last_position_deg = pos;
        Ok(delta)
    }

    /// Command a target angular speed in degrees/second via duty-cycle
    /// direct drive (no built-in speed regulation -- matches the `run_direct`
    /// + `set_duty_cycle_sp` pattern from ev3dev-lang-rust's own example).
    pub fn set_speed_deg_s(&self, target_deg_s: i32, max_deg_s: i32) -> Ev3Result<()> {
        let duty = ((target_deg_s as i64 * 100) / max_deg_s.max(1) as i64).clamp(-100, 100) as i32;
        self.motor.set_duty_cycle_sp(duty)
    }
}

pub struct KickerMotor {
    motor: LargeMotor,
}

impl KickerMotor {
    pub fn new(port: MotorPort) -> Ev3Result<Self> {
        let motor = LargeMotor::get(port)?;
        motor.run_direct()?;
        Ok(Self { motor })
    }

    pub fn fire_kick(&self, duty: i32) -> Ev3Result<()> {
        self.motor.set_duty_cycle_sp(duty.clamp(0, 100))
    }

    pub fn stop(&self) -> Ev3Result<()> {
        self.motor.set_duty_cycle_sp(0)
    }
}
