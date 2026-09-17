/// Benchmarks the gyro sensor.

use std::time::{Duration, Instant};

use ev3_balance_bot_firmware::sensors::gyro::Gyro;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gyro = Gyro::find().map_err(|error| {
        eprintln!("gyro initialization failed: {error:?}");
        error
    })?;
    let start = Instant::now();
    let mut count = 0;
    let mut unique_reads = 0;
    let (mut angle, mut rate) = (0.0, 0.0);
    while start.elapsed() < Duration::from_secs(5) {
        let (new_angle, new_rate) = gyro.sample().map_err(|error| {
            eprintln!("gyro angle sample failed: {error:?}");
            error
        })?;
        if new_rate != rate {
            unique_reads += 1;
        }
        angle = new_angle;
        rate = new_rate;
        count += 1;
    }
    println!("Read {} times in 5 seconds, with {} unique rate readings", count, unique_reads);
    println!("Last read angle: {}", angle);
    println!("Last read rate: {}", rate);
    println!("Benchmark completed.");

    std::thread::sleep(Duration::from_secs(3));
    
    Ok(())
}