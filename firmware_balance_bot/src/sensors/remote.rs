//! Wrap `ev3dev_lang_rust::sensors::{InfraredSensor, RemoteControl}` into
//! a two-wheel continuous control signal, mirroring
//! `balance_bot.tasks.mdp.RemoteControlCommand`'s `command` property (sim
//! side samples continuous `[-1, 1]` values; the real EV3 remote only has
//! up/down/released per slider, i.e. `{-1, 0, 1}`, which is within that
//! trained range).
//!
//! Verified against the installed crate source
//! (`ev3dev-lang-rust/src/sensors/infrared_sensor.rs`,
//! `examples/infrared-sensor.rs`): one `InfraredSensor` in `IR-REMOTE` mode
//! exposes both the red (left) and blue (right) sliders of one channel via
//! `RemoteControl::is_red_up/down` / `is_blue_up/down`.

use ev3dev_lang_rust::sensors::{InfraredSensor, RemoteControl};
use ev3dev_lang_rust::Ev3Result;

pub struct Remote {
    remote: RemoteControl,
}

impl Remote {
    pub fn find(channel: u8) -> Ev3Result<Self> {
        let sensor = InfraredSensor::find()?;
        let remote = RemoteControl::new(sensor, channel)?;
        Ok(Self { remote })
    }

    /// Left (red) / right (blue) wheel control, each in `{-1.0, 0.0, 1.0}`.
    pub fn wheel_controls(&self) -> Ev3Result<[f32; 2]> {
        self.remote.process()?;

        let left = if self.remote.is_red_up() {
            1.0
        } else if self.remote.is_red_down() {
            -1.0
        } else {
            0.0
        };
        let right = if self.remote.is_blue_up() {
            1.0
        } else if self.remote.is_blue_down() {
            -1.0
        } else {
            0.0
        };
        Ok([left, right])
    }
}
