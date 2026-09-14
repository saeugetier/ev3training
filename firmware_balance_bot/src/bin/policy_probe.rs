//! Deterministic host-side probe for the deployed policy inference.
//!
//! This tests `policy::infer` without EV3 sensors or motors. Inputs are Q15
//! values at `policy_weights::OBS_SCALE`, exactly as the firmware receives.
//!
//! Examples:
//!   cargo run --bin policy_probe -- --pattern zero
//!   cargo run --bin policy_probe -- --pattern constant --value 1000
//!   cargo run --bin policy_probe -- --pattern impulse --index 0 --value 1000
//!   cargo run --bin policy_probe -- 0 0 0 ...  (exactly 36 values)

use ev3_balance_bot_firmware::config::INPUT_DIM;
use ev3_balance_bot_firmware::policy;
use ev3_balance_bot_firmware::policy_weights::ACTION_SCALE;

fn print_result(label: &str, obs: &[i16; INPUT_DIM]) {
    let trace = policy::infer_trace(obs);
    let action = trace.action;
    println!("test: {label}");
    println!("  obs_q15: {:?}", obs);
    println!("  action_q15: {:?}", action);
    println!(
        "  action_real: left={:.6} right={:.6}",
        action[0] as f32 * ACTION_SCALE,
        action[1] as f32 * ACTION_SCALE,
    );
    println!(
        "  action_clamped: left={:.6} right={:.6}",
        (action[0] as f32 * ACTION_SCALE).clamp(-1.0, 1.0),
        (action[1] as f32 * ACTION_SCALE).clamp(-1.0, 1.0),
    );
    println!("  trace_fc1_first8: {:?}", &trace.fc1[..8]);
    println!("  trace_fc2_first8: {:?}", &trace.fc2[..8]);
    println!("  trace_fc3_first8: {:?}", &trace.fc3[..8]);
}

fn parse_value(value: Option<&String>, name: &str) -> i16 {
    value
        .unwrap_or_else(|| panic!("missing value for {name}"))
        .parse::<i16>()
        .unwrap_or_else(|_| panic!("{name} must be an i16 Q15 value"))
}

fn parse_explicit(values: &[String]) -> [i16; INPUT_DIM] {
    if values.len() != INPUT_DIM {
        panic!(
            "expected exactly {INPUT_DIM} Q15 values, received {}",
            values.len()
        );
    }
    std::array::from_fn(|index| {
        values[index]
            .parse::<i16>()
            .unwrap_or_else(|_| panic!("argument {} is not an i16", index + 1))
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: policy_probe --pattern zero|constant|impulse [options]");
        eprintln!("   or: policy_probe <exactly 36 Q15 values>");
        std::process::exit(2);
    }

    if args[0] != "--pattern" {
        let obs = parse_explicit(&args);
        print_result("explicit", &obs);
        return;
    }

    let pattern = args.get(1).map(String::as_str).unwrap_or("");
    let mut obs = [0i16; INPUT_DIM];
    match pattern {
        "zero" => print_result("zero", &obs),
        "constant" => {
            let value = parse_value(args.get(3), "--value");
            obs.fill(value);
            print_result("constant", &obs);
        }
        "impulse" => {
            let index = args
                .get(3)
                .unwrap_or_else(|| panic!("missing value for --index"))
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("--index must be an integer"));
            if index >= INPUT_DIM {
                panic!("--index must be in [0, {})", INPUT_DIM - 1);
            }
            obs[index] = parse_value(args.get(5), "--value");
            print_result("impulse", &obs);
        }
        _ => panic!("unknown pattern '{pattern}'"),
    }
}
