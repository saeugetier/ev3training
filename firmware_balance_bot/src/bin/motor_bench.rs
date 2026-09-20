// Motor bench testing for the balance bot firmware.
// This binary is used to test the motors of the balance bot independently of the main firmware.
// It allows testing how fast and accurately the motors respond to commands without involving the full balance bot control loop.
use std::time::Duration;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use ctrlc;

use ev3_balance_bot_firmware::sensors::tacho::DriveMotor;
use ev3dev_lang_rust::motors::MotorPort;

fn main() -> ev3dev_lang_rust::Ev3Result<()> {
    let right_motor = DriveMotor::new(MotorPort::OutA).map_err(|error| {
        eprintln!("right motor initialization failed on OutA: {error:?}");
        error
    })?;

    let stop_requested = Arc::new(AtomicBool::new(false));
    let stop_requested_handler = Arc::clone(&stop_requested);
    ctrlc::set_handler(move || {
        eprintln!("Ctrl-C signal received");
        stop_requested_handler.store(true, Ordering::SeqCst);
    })
    .map_err(|error| {
        eprintln!("failed to install Ctrl-C handler: {error}");
        ev3dev_lang_rust::Ev3Error::InternalError {
            msg: format!("failed to install Ctrl-C handler: {error}"),
        }
    })?;


    let mut effort: f32 = 0.0; // Set the motor effort to 50%
    let mut direction: f32 = 0.1; // 1.0 for increasing effort, -1.0 for decreasing effort
    loop {
        if stop_requested.load(Ordering::SeqCst) {
            eprintln!("Ctrl-C received, stopping motors");
            right_motor.stop()?;
            return Ok(());
        }

        let _ = right_motor.set_effort(effort); // Run the right motor at 50% power indefinitely
        std::thread::sleep(Duration::from_millis(10)); 

        // increment and decrement the effort for testing purposes from -1.0 to 1.0
        effort += direction; // Increment the effort for testing purposes
        if effort >= 1.0 {
            effort = 1.0;
            direction = -direction;
        } else if effort <= -1.0 {
            effort = -1.0;
            direction = -direction;
        }
    }

}