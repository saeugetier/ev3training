//! Dummy benchmark target: proves the quantized policy forward pass can
//! run within the 50 Hz control budget (`config::CONTROL_PERIOD_MS`).
//!
//! Uses synthetic (pseudo-random, deterministic) observations instead of
//! real sensors -- this only measures NN inference latency, not I2C/sysfs
//! sensor I/O.
//!
//! IMPORTANT: host x86_64 timing is NOT representative of the real EV3
//! ARMv5TEJ CPU (no FPU, ~300MHz). Cross-compile and run this binary
//! on-target (or under `qemu-arm-static` as an ARM-instruction proxy) for
//! a meaningful result:
//!
//! ```bash
//! cargo build --release --target armv5te-unknown-linux-musleabi --bin policy_bench
//! qemu-arm-static target/armv5te-unknown-linux-musleabi/release/policy_bench
//! # or copy the binary to the EV3 and run it there directly.
//! ```

use std::time::{Duration, Instant};

use ev3_robocup_firmware::config::{CONTROL_PERIOD_MS, INPUT_DIM};
use ev3_robocup_firmware::policy::{self, PolicyState};

const WARMUP_ITERS: usize = 50;
const BENCH_ITERS: usize = 2000;

/// Minimal deterministic xorshift PRNG -- avoids pulling in the `rand`
/// crate just for synthetic benchmark inputs.
struct Xorshift32(u32);

impl Xorshift32 {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Next value in roughly [-4096, 4096], a plausible Q15 observation range.
    fn next_i16(&mut self) -> i16 {
        (self.next_u32() % 8192) as i16 - 4096
    }
}

fn make_observation(rng: &mut Xorshift32) -> [i16; INPUT_DIM] {
    let mut obs = [0i16; INPUT_DIM];
    for v in obs.iter_mut() {
        *v = rng.next_i16();
    }
    obs
}

fn main() {
    let mut rng = Xorshift32(0xC0FFEE_u32);
    let mut state = PolicyState::default();
    let budget = Duration::from_millis(CONTROL_PERIOD_MS);

    // Warm up: first calls can be slower (cold caches, lazy page faults).
    for _ in 0..WARMUP_ITERS {
        let obs = make_observation(&mut rng);
        let _ = policy::step(&obs, &mut state);
    }

    let mut durations = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let obs = make_observation(&mut rng);
        let start = Instant::now();
        let _action = std::hint::black_box(policy::step(&obs, &mut state));
        durations.push(start.elapsed());
    }

    durations.sort();
    let total: Duration = durations.iter().sum();
    let mean = total / BENCH_ITERS as u32;
    let p50 = durations[BENCH_ITERS / 2];
    let p99 = durations[(BENCH_ITERS * 99) / 100];
    let worst = *durations.last().unwrap();

    println!("Policy inference benchmark ({BENCH_ITERS} iterations, {WARMUP_ITERS} warmup)");
    println!("  control budget (50 Hz): {budget:?}");
    println!("  mean:  {mean:?}");
    println!("  p50:   {p50:?}");
    println!("  p99:   {p99:?}");
    println!("  worst: {worst:?}");

    let pass = worst < budget;
    println!(
        "  result: {} (worst-case {} budget)",
        if pass { "PASS" } else { "FAIL" },
        if pass { "within" } else { "exceeds" }
    );

    if !pass {
        std::process::exit(1);
    }
}
