// displays the current angle and rate of the gyro sensor on screen

use std::time::{Duration, Instant};

use ev3_balance_bot_firmware::sensors::gyro::Gyro;
use ev3_balance_bot_firmware::display::Display;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gyro = Gyro::find().map_err(|error| {
        eprintln!("gyro initialization failed: {error:?}");
        error
    })?;
    let start = Instant::now();

    let mut display = Display::new().map_err(|error| {
        eprintln!("display initialization failed: {error:?}");
        error
    })?;
    
    while start.elapsed() < Duration::from_secs(60) {
        let (new_angle, new_rate) = gyro.sample().map_err(|error| {
            eprintln!("gyro angle sample failed: {error:?}");
            error
        })?;
        //println!("Current angle: {}, Current rate: {}", angle, rate);

        display.show_gyro(new_angle, new_rate);
        std::thread::sleep(Duration::from_millis(100));
    }

    Ok(())
}