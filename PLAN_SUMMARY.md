# Project Plan Summary

> English summary of the working plan kept in session memory
> (`/memories/session/plan.md`). See that file for the full, evolving log.

## Goal

Train a RoboCup-style soccer policy in simulation and deploy it for
real-time inference on a LEGO EV3 brick (ARMv5TEJ, no FPU), running Rust
firmware built on `ev3dev-lang-rust` and `embedded-nn`, with the network
quantized to Q15 (int16) fixed point.

## Key confirmed facts and decisions

- **Simulator**: [mjlab](https://github.com/mujocolab/mjlab) (Isaac-Lab-style
  manager-based API on MuJoCo Warp). Training needs an NVIDIA GPU;
  evaluation-only on macOS.
- **Task**: 1v0 — dribble the ball to the opponent's goal and kick it in.
- **Sensors modeled**: VL53L8CX 8x8 depth grid (64 zones, no downsampling),
  gyro (heading + yaw rate), wheel motor relative-rotation feedback.
- **Actions**: left/right wheel speed + a kick trigger (all continuous,
  tanh-bounded), the kick trigger is thresholded on-device to fire the
  kicker motor.
- **Policy core**: recurrent (LSTM), required for memory/partial
  observability — a pure feed-forward MLP was ruled insufficient.
- **`ev3dev-lang-rust`** has `GyroSensor` and `TachoMotor` support, but *no*
  built-in VL53L8CX/generic-I2C sensor — the depth sensor is driven via raw
  Linux I2C (`i2cdev` crate against `/dev/i2c-*`), bypassing the
  ev3dev sensor abstraction.
- **`embedded-nn` v0.2.1** has no ONNX/TFLite import path for LSTMs, so the
  project builds its own weight-export/quantization pipeline and calls
  `embedded-nn`'s low-level kernels (`fully_connected_s16`, `sigmoid_s16`,
  `tanh_s16`, `elementwise_mul_s16`, `elementwise_add_s16`) directly rather
  than relying on any import/compiler crate.
- **No FPU (ARMv5TEJ)**: all inference math is pure fixed-point
  (multiply + shift), verified to require no floating-point instructions on
  the hot path.

## Empirically verified `embedded-nn` v0.2.1 API facts

These were confirmed with a disposable scratch Cargo project rather than
assumed from documentation, and are the basis of the whole quantization
pipeline:

- `fully_connected_s16` requires **int8 weights** and **int64 bias**
  (only activations are int16).
- `Dims` convention for the filter tensor is `Dims::new(IN, 1, 1, OUT)`
  (`n` = input dim, `c` = output dim — the reverse of what the field names
  suggest).
- The kernel's internal accumulator does an extra `>> 15` before
  requantizing, which must be compensated for in the multiplier
  calculation (`real_multiplier *= 32768`).
- `sigmoid_s16`/`tanh_s16` expect input in a Q3.12 fixed-point format at
  `left_shift = 0`; output is Q0.15. The preceding layer's output scale is
  chosen to be an exact power of two so no extra rounding error is
  introduced.
- `elementwise_add_s16` supports two independently-scaled inputs
  (`mult1/shift1`, `mult2/shift2`), which is used to combine a gate's
  `Wx @ obs` and `Wh @ h` contributions without needing to concatenate
  tensors living at different scales.

See `/memories/embedded-nn-crate-facts.md` for the durable version of this
note.

## Implementation status

| Area | Status |
|---|---|
| `sim/mjlab_robocup/` (MJCF robot, task config, rewards, PPO cfg) | Implemented, **not yet run against a real mjlab install** — some field/method names are best-effort and flagged with TODOs. |
| `tools/quantize_export/` (calibration + Q15 export pipeline) | Implemented and verified end-to-end with a random-weight smoke test. |
| `firmware/` (Rust inference + control loop) | Implemented, compiles and passes unit tests on host and on the real `armv5te-unknown-linux-musleabi` target; benchmark verified under `qemu-arm-static`. |
| VL53L8CX ranging driver | Only the I2C **platform layer** (register read/write) is implemented; the actual ST ULD ranging protocol is not reimplemented (recommended: FFI-bind ST's official C driver). |
| Training a real checkpoint | Not done — `firmware/src/policy_weights.rs` is an all-zero placeholder. |

## Architecture evolution: network size

The deployed network was redesigned mid-project to reduce its parameter
count:

- **Original**: Dense encoder (72→64) → concat with hidden state → 4 LSTM
  gates operating on a 128-dim input → Dense output. **37,891 parameters**.
- **Final ("Variant A")**: LSTM operates directly on the raw 72-dim
  observation (hidden size 32) → Dense+Tanh decoder (32) → Dense+Tanh
  action head. **14,595 parameters** (~2.6x smaller), and it matches
  `rsl_rl`'s stock `ActorCriticRecurrent` structure exactly (no custom
  training code needed).

See `docs/NEURAL_NETWORK.md` for the full architecture description.

## Next steps for a real deployment

1. Install mjlab and verify the scene/entity/action-term API surface used
   in `sim/mjlab_robocup/`.
2. Train the policy and confirm `rsl_rl`'s `ActorCriticRecurrent` matches
   the config in `agents/rsl_rl_ppo_cfg.py`.
3. Record a rollout, run `tools/quantize_export/cli.py` against the real
   checkpoint, and replace the placeholder `policy_weights.rs`.
4. Port or FFI-bind a real VL53L8CX ranging driver.
5. Validate on physical EV3 hardware (sensor sanity checks, then
   closed-loop control).
