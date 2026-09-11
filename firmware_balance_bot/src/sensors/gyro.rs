//! Thin wrapper around `ev3dev_lang_rust::sensors::GyroSensor`, in combined
//! Angle + Rotational Speed mode (`GYRO-G&A`), verified against the
//! installed crate source (`ev3dev-lang-rust/src/sensors/gyro_sensor.rs`):
//! `get_angle()`/`get_rotational_speed()` both read from that one mode
//! (value0 / value1) without a separate mode switch per call.

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

    /// Tilt angle about the wheel axis [rad]. Zeroed by the sensor at
    /// power-up, matching the sim's `imu_pitch` (see `mdp.py`).
    pub fn angle_rad(&self) -> Ev3Result<f32> {
        Ok((self.sensor.get_angle()? as f32).to_radians())
    }

    /// Tilt rate about the wheel axis [rad/s].
    pub fn rate_rad_s(&self) -> Ev3Result<f32> {
        Ok((self.sensor.get_rotational_speed()? as f32).to_radians())
    }
}
