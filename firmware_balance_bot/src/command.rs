//! Mirrors `RemoteControlCommand`'s odometry integrator in
//! `balance_robot_rl/src/balance_bot/tasks/mdp.py` (`_update_command`),
//! so the on-device `wheel_distance_error` observation matches training.

use crate::config::{MAX_POS_ERROR_RAD, WHEEL_SPEED_RAD_S};

pub struct RemoteCommandState {
    ref_wheel_pos_rad: [f32; 2],
}

impl RemoteCommandState {
    /// `measured_wheel_pos_rad` seeds the reference at the current wheel
    /// position, matching `_resample_command`'s reset-on-episode-start
    /// behavior (the real robot has one continuous "episode").
    pub fn new(measured_wheel_pos_rad: [f32; 2]) -> Self {
        Self {
            ref_wheel_pos_rad: measured_wheel_pos_rad,
        }
    }

    /// Integrate the commanded wheel rotation, anti-windup clamp it
    /// against the measured position, and return the saturated odometry
    /// error `[rad]` per wheel (same value as `wheel_pos_error` in sim).
    pub fn update(&mut self, controls: [f32; 2], measured_wheel_pos_rad: [f32; 2]) -> [f32; 2] {
        let mut error = [0.0f32; 2];
        for i in 0..2 {
            let target_vel_rad_s = controls[i] * WHEEL_SPEED_RAD_S;
            self.ref_wheel_pos_rad[i] += target_vel_rad_s * crate::config::STEP_DT_S;

            let diff = (self.ref_wheel_pos_rad[i] - measured_wheel_pos_rad[i])
                .clamp(-MAX_POS_ERROR_RAD, MAX_POS_ERROR_RAD);
            self.ref_wheel_pos_rad[i] = measured_wheel_pos_rad[i] + diff;
            error[i] = diff;
        }
        error
    }
}
