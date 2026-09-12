//! Thin wrapper around `ev3dev_lang_rust::sensors::GyroSensor`.
//!
//! The EV3 driver advertises `GYRO-G&A`, but this sensor times out when that
//! mode is selected. The firmware therefore uses the working `GYRO-ANG` mode
//! and differentiates consecutive angle samples for rotational speed.

use ev3dev_lang_rust::sensors::GyroSensor;
use ev3dev_lang_rust::Ev3Result;
use std::time::Instant;

pub struct Gyro {
    sensor: GyroSensor,
    last_angle_rad: Option<f32>,
    last_sample: Instant,
}

impl Gyro {
    pub fn find() -> Ev3Result<Self> {
        let sensor = GyroSensor::find()?;
        sensor.set_mode_gyro_ang()?;
        Ok(Self {
            sensor,
            last_angle_rad: None,
            last_sample: Instant::now(),
        })
    }

    /// Read tilt angle and derive tilt rate from consecutive angle samples.
    pub fn sample(&mut self) -> Ev3Result<(f32, f32)> {
        let angle_rad = (self.sensor.get_angle()? as f32).to_radians();
        let now = Instant::now();
        let dt = now.duration_since(self.last_sample).as_secs_f32();
        let rate_rad_s = self
            .last_angle_rad
            .map(|last| (angle_rad - last) / dt.max(1e-4))
            .unwrap_or(0.0);
        self.last_angle_rad = Some(angle_rad);
        self.last_sample = now;
        Ok((angle_rad, rate_rad_s))
    }
}
