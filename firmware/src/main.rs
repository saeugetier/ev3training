use std::thread;
use std::time::Duration;

use ev3dev_lang_rust::motors::MotorPort;
use ev3dev_lang_rust::Ev3Result;

use ev3_robocup_firmware::sensors::gyro::Gyro;
use ev3_robocup_firmware::sensors::tacho::{DriveMotor, KickerMotor};
use ev3_robocup_firmware::sensors::vl53l8cx::{
    DepthFrame, Vl53l8cx, Vl53l8cxPlatform, VL53L8CX_DEFAULT_ADDRESS,
};
use ev3_robocup_firmware::{actuation, config, observation, policy};

// TODO: verify the actual /dev/i2c-N node for whichever EV3 input port the
// VL53L8CX is wired to (depends on ev3dev's lego-port/nxt-i2c-sensor setup).
const VL53L8CX_I2C_BUS: &str = "/dev/i2c-3";

fn main() -> Ev3Result<()> {
    let gyro = Gyro::find()?;
    let mut left_motor = DriveMotor::new(MotorPort::OutA)?;
    let mut right_motor = DriveMotor::new(MotorPort::OutB)?;
    let kicker_motor = KickerMotor::new(MotorPort::OutC)?;

    let depth_platform = Vl53l8cxPlatform::new(VL53L8CX_I2C_BUS, VL53L8CX_DEFAULT_ADDRESS)
        .expect("failed to open VL53L8CX I2C bus");
    let mut depth_sensor = Vl53l8cx::new(depth_platform);
    let _ = depth_sensor.platform_mut(); // ranging not yet implemented, see sensors/vl53l8cx.rs

    let mut policy_state = policy::PolicyState::default();
    let mut last_action = [0i16; 3];

    loop {
        let tick_start = std::time::Instant::now();

        // TODO: replace with depth_sensor.read_frame() once the ULD ranging
        // driver is wired up (see sensors/vl53l8cx.rs module docs).
        let depth = DepthFrame::default();

        let yaw_deg = gyro.angle_deg()?;
        let yaw_rate_deg_s = gyro.rate_deg_s()?;
        let left_delta_deg = left_motor.poll_delta_deg()?;
        let right_delta_deg = right_motor.poll_delta_deg()?;

        let obs = observation::build_observation(
            &depth,
            yaw_deg,
            yaw_rate_deg_s,
            left_delta_deg,
            right_delta_deg,
            &last_action,
        );

        let action = policy::step(&obs, &mut policy_state);
        actuation::apply_action(&action, &left_motor, &right_motor, &kicker_motor)?;
        last_action = action;

        let elapsed = tick_start.elapsed();
        let period = Duration::from_millis(config::CONTROL_PERIOD_MS);
        if elapsed < period {
            thread::sleep(period - elapsed);
        }
    }
}
