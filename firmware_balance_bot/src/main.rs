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
    let mut gyro = Gyro::find().map_err(|error| {
        eprintln!("gyro initialization failed: {error:?}");
        error
    })?;
    let remote = Remote::find(REMOTE_CHANNEL).map_err(|error| {
        eprintln!("IR remote initialization failed on channel {REMOTE_CHANNEL}: {error:?}");
        error
    })?;
    let left_motor = DriveMotor::new(MotorPort::OutA).map_err(|error| {
        eprintln!("left motor initialization failed on OutA: {error:?}");
        error
    })?;
    let right_motor = DriveMotor::new(MotorPort::OutB).map_err(|error| {
        eprintln!("right motor initialization failed on OutB: {error:?}");
        error
    })?;

    let left_position = left_motor.position_rad().map_err(|error| {
        eprintln!("left motor position read failed: {error:?}");
        error
    })?;
    let right_position = right_motor.position_rad().map_err(|error| {
        eprintln!("right motor position read failed: {error:?}");
        error
    })?;
    let mut command_state = RemoteCommandState::new([left_position, right_position]);
    let mut history = ObservationHistory::new();
    let mut last_action = [0.0f32; 2];

    loop {
        let tick_start = std::time::Instant::now();

        let (gyro_angle_rad, gyro_tilt_rate_rad_s) = gyro.sample().map_err(|error| {
            eprintln!("gyro angle sample failed: {error:?}");
            error
        })?;
        let left_pos_rad = left_motor.position_rad().map_err(|error| {
            eprintln!("left motor position read failed: {error:?}");
            error
        })?;
        let right_pos_rad = right_motor.position_rad().map_err(|error| {
            eprintln!("right motor position read failed: {error:?}");
            error
        })?;
        let wheel_speed_rad_s = [
            left_motor.speed_rad_s().map_err(|error| {
                eprintln!("left motor speed read failed: {error:?}");
                error
            })?,
            right_motor.speed_rad_s().map_err(|error| {
                eprintln!("right motor speed read failed: {error:?}");
                error
            })?,
        ];
        let remote_controls = remote.wheel_controls().map_err(|error| {
            eprintln!("IR remote read failed: {error:?}");
            error
        })?;

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
        last_action = actuation::apply_action(&action, &left_motor, &right_motor).map_err(|error| {
            eprintln!("motor duty-cycle write failed: {error:?}");
            error
        })?;

        let elapsed = tick_start.elapsed();
        let period = Duration::from_millis(config::CONTROL_PERIOD_MS);
        if elapsed < period {
            thread::sleep(period - elapsed);
        }
    }
}
