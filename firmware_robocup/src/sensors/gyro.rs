//! Thin wrapper around `ev3dev_lang_rust::sensors::GyroSensor`.
//!
//! Mode-setter/getter method names below are best-effort based on
//! ev3dev-lang-rust's `sensor_mode!` macro convention -- verify exact
//! names against the installed crate version's docs.rs page.

use ev3dev_lang_rust::sensors::GyroSensor;
use ev3dev_lang_rust::Ev3Result;

pub struct Gyro {
    sensor: GyroSensor,
}

impl Gyro {
    pub fn find() -> Ev3Result<Self> {
        let sensor = GyroSensor::find()?;
        sensor.set_mode_gyro_g_and_a()?; // Combined angle + rate mode.
        Ok(Self { sensor })
    }

    /// Heading angle in degrees (unbounded, accumulates drift over time).
    pub fn angle_deg(&self) -> Ev3Result<i32> {
        self.sensor.get_angle()
    }

    /// Angular rate in degrees/second about the vertical axis.
    pub fn rate_deg_s(&self) -> Ev3Result<i32> {
        self.sensor.get_rotational_speed()
    }
}
