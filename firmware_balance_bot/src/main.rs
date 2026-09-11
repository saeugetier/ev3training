use std::thread;
use std::time::Duration;

use ev3dev_lang_rust::motors::MotorPort;
use ev3dev_lang_rust::Ev3Result;

use ev3_balance_bot_firmware::command::RemoteCommandState;
use ev3_balance_bot_firmware::observation::{build_step_obs, quantize_obs, ObservationHistory};
use ev3_balance_bot_firmware::sensors::gyro::Gyro;
use ev3_balance_bot_firmware::sensors::remote::Remote;
use ev3_balance_bot_firmware::sensors::tacho::DriveMotor;
use ev3_balance_bot_firmware::{actuation, config, policy};

// TODO: verify which of the 4 remote channels the physical EV3 remote is
// paired to; channel 1 is ev3dev-lang-rust's own example default.
const REMOTE_CHANNEL: u8 = 1;

fn main() -> Ev3Result<()> {
    let gyro = Gyro::find()?;
    let remote = Remote::find(REMOTE_CHANNEL)?;
    let left_motor = DriveMotor::new(MotorPort::OutA)?;
    let right_motor = DriveMotor::new(MotorPort::OutB)?;

    let mut command_state = RemoteCommandState::new([
        left_motor.position_rad()?,
        right_motor.position_rad()?,
    ]);
    let mut history = ObservationHistory::new();
    let mut last_action = [0.0f32; 2];

    loop {
        let tick_start = std::time::Instant::now();

        let gyro_angle_rad = gyro.angle_rad()?;
        let gyro_tilt_rate_rad_s = gyro.rate_rad_s()?;
        let left_pos_rad = left_motor.position_rad()?;
        let right_pos_rad = right_motor.position_rad()?;
        let wheel_speed_rad_s = [left_motor.speed_rad_s()?, right_motor.speed_rad_s()?];
        let remote_controls = remote.wheel_controls()?;

        let wheel_distance_error_rad =
            command_state.update(remote_controls, [left_pos_rad, right_pos_rad]);

        let step_obs = build_step_obs(
            gyro_angle_rad,
            gyro_tilt_rate_rad_s,
            wheel_speed_rad_s,
            wheel_distance_error_rad,
            remote_controls,
            last_action,
        );
        history.push(step_obs);

        let obs = quantize_obs(&history.flatten());
        let action = policy::infer(&obs);
        last_action = actuation::apply_action(&action, &left_motor, &right_motor)?;

        let elapsed = tick_start.elapsed();
        let period = Duration::from_millis(config::CONTROL_PERIOD_MS);
        if elapsed < period {
            thread::sleep(period - elapsed);
        }
    }
}
