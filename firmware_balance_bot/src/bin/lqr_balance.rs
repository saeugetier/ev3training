use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

use ev3dev_lang_rust::motors::MotorPort;
use ev3dev_lang_rust::Ev3Result;

use ev3_balance_bot_firmware::{command::RemoteCommandState, sensors::{gyro, remote}};
use ev3_balance_bot_firmware::display::Display;
use ev3_balance_bot_firmware::sensors::gyro::Gyro;
use ev3_balance_bot_firmware::sensors::remote::Remote;
use ev3_balance_bot_firmware::sensors::tacho::DriveMotor;
use ev3_balance_bot_firmware::lqr::BalanceBot;

fn main() -> Ev3Result<()> {
    // Your main function implementation here
    let dt = 0.005; // 5 ms update interval
    let mut bot = BalanceBot::new(dt);

    let remote = Remote::find(1).unwrap();
    let mut gyro = Gyro::find().unwrap();
    let left_motor = DriveMotor::new(MotorPort::OutA).unwrap();
    let right_motor = DriveMotor::new(MotorPort::OutD).unwrap();

    bot.set_velocity(0.0);

    // Geradeaus
    bot.set_turn(0.0);

    loop {
        let gyro_sample = gyro.sample().unwrap();
        let command = remote.wheel_controls().unwrap();
        bot.set_velocity(((command[0] + command[1]) * 0.2).into());
        bot.set_turn(((command[1] - command[0]) * 0.2).into());
        // Hier würden die Sensordaten gelesen werden
        let left_position = left_motor.position_rad().unwrap();
        let right_position = right_motor.position_rad().unwrap();
        let position = (left_position + right_position) / 2.0;
        let velocity = (left_motor.speed_rad_s().unwrap() + right_motor.speed_rad_s().unwrap()) / 2.0;
        let angle = gyro_sample.0;
        let angular_velocity = gyro_sample.1;

        let (left, right) = bot.update(position.into(), velocity.into(), angle.into(), angular_velocity.into());

        // Hier würden die Motoren angesteuert werden
        println!("Left motor: {}, Right motor: {}", left, right);

        left_motor.set_effort(left as f32).unwrap();
        right_motor.set_effort(right as f32).unwrap();

        thread::sleep(Duration::from_millis((dt * 1000.0) as u64));
    }


    Ok(())
}
