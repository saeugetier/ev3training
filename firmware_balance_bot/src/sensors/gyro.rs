//! Thin wrapper around `ev3dev_lang_rust::sensors::GyroSensor`.
//!
//! The EV3 driver advertises `GYRO-G&A`, but this sensor times out when that
//! mode is selected. The firmware therefore uses the working `GYRO-RATE` mode
//! and integrates the native rotational speed to estimate the angle.

use ev3dev_lang_rust::sensors::GyroSensor;
use ev3dev_lang_rust::Ev3Result;
use std::time::Instant;

const PI: f32 = std::f32::consts::PI;
const TWO_PI: f32 = 2.0 * PI;

pub struct Gyro {
    sensor: GyroSensor,
    angle_rad: f32,
    last_sample: Instant,
}

impl Gyro {
    pub fn find() -> Ev3Result<Self> {
        let sensor = GyroSensor::find()?;
        sensor.set_mode_gyro_ang()?;
        Ok(Self {
            sensor,
            angle_rad: 0.0,
            last_sample: Instant::now(),
        })
    }

    /// Read native rotational speed and integrate it to estimate tilt angle.
    pub fn sample(&mut self) -> Ev3Result<(f32, f32)> {
        let current_angle_rad = (self.sensor.get_angle()? as f32).to_radians();
        let now = Instant::now();
        let dt = now.duration_since(self.last_sample).as_secs_f32();
        // Clamp the time step to avoid large jumps in the estimated angle.
        let dt = dt.min(0.1);
        let rate_rad_s = (current_angle_rad - self.angle_rad) / dt;
 
        self.angle_rad = (current_angle_rad + PI).rem_euclid(TWO_PI) - PI;
        self.last_sample = now;
        Ok((self.angle_rad, rate_rad_s))
    }
}
