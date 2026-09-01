# EV3 RoboCup Policy: mjlab → Q15 → embedded-nn

A small recurrent RL policy that plays 1-robot RoboCup soccer (dribble the
ball to the goal and kick it in), trained in simulation with
[mjlab](https://github.com/mujocolab/mjlab) and deployed for real-time,
integer-only inference on a LEGO **EV3** brick (ARMv5TEJ, no FPU) using
Rust, [`ev3dev-lang-rust`](https://github.com/ev3dev/ev3dev-lang-rust) and
[`embedded-nn`](https://crates.io/crates/embedded-nn).

For the full background and design decisions behind this project, see
[`PLAN_SUMMARY.md`](PLAN_SUMMARY.md). For a detailed description of the
network itself, see [`docs/NEURAL_NETWORK.md`](docs/NEURAL_NETWORK.md).

## Intention

Real EV3 hardware has no floating-point unit and very limited flash/RAM,
so a policy trained with standard deep-RL tooling (PyTorch, GPU) cannot
run as-is on the robot. This project bridges that gap end to end:

1. **Simulate & train** a robot that closely matches the physical EV3
   build (same sensors, same control-loop rate, same action space) so the
   trained policy transfers to hardware with minimal surprises.
2. **Quantize** the trained network to Q15 (int16) fixed point, with all
   the math worked out so it needs no floating-point instructions at
   inference time.
3. **Run** the quantized network on-device using `embedded-nn`'s
   low-level integer kernels, driven by a Rust control loop that reads
   the real sensors and commands the real motors at 50 Hz.

## Repository layout

```
sim/mjlab_robocup/        mjlab task: robot MJCF, observations/actions/
                          rewards, PPO training config, task registration.
tools/quantize_export/    Calibrates a trained checkpoint and exports
                          Q15 (int8 weight / int16 activation) weights
                          as Rust source, for the firmware to embed.
firmware/                 Rust crate: sensor drivers, quantized inference,
                          motor control, control loop, benchmark binary.
docs/NEURAL_NETWORK.md    Detailed description of the deployed network.
PLAN_SUMMARY.md           Design background, decisions and open items.
```

## Hardware

- LEGO EV3 brick (ARMv5TEJ CPU, no FPU, runs [ev3dev](https://www.ev3dev.org/)).
- 2 drive motors (differential drive) + 1 kicker motor.
- EV3 gyro sensor (heading + yaw rate).
- A **VL53L8CX** ToF depth sensor (8x8 zone ranging) wired over I2C —
  not a stock EV3 sensor, so it's driven directly over Linux I2C rather
  than through `ev3dev-lang-rust`'s sensor abstraction.

## Build process

### 1. Simulation / training environment (Python)

```bash
# Requires an NVIDIA GPU (mjlab training is not supported on CPU/macOS).
git clone https://github.com/mujocolab/mjlab.git
cd mjlab && uv sync
cd /path/to/sim && uv pip install -e .   # installs mjlab_robocup as a package
```

### 2. Firmware (Rust, cross-compiled for the EV3)

The firmware targets `armv5te-unknown-linux-musleabi` (a static-musl
build, so no external glibc ARM toolchain is required):

```bash
cd firmware
rustup target add armv5te-unknown-linux-musleabi
cargo build --release --target armv5te-unknown-linux-musleabi
```

The resulting `target/armv5te-unknown-linux-musleabi/release/ev3-robocup`
binary can be copied straight to the EV3 (e.g. via `scp`) and run there —
it's statically linked, so no runtime dependencies need to be installed
on-device.

To sanity-check the binary without hardware, run it under ARM user-mode
emulation:

```bash
qemu-arm-static target/armv5te-unknown-linux-musleabi/release/ev3-robocup
```

### 3. Performance verification

A dedicated benchmark binary proves the quantized network fits inside the
50 Hz (20 ms) control budget:

```bash
cargo build --release --target armv5te-unknown-linux-musleabi --bin policy_bench
qemu-arm-static target/armv5te-unknown-linux-musleabi/release/policy_bench
# or copy to the EV3 and run natively for a hardware-accurate measurement.
```

## Training process

### Prerequisites

- An NVIDIA GPU with enough VRAM for the chosen `num-envs` (mjlab/MuJoCo
  Warp parallelizes environments on the GPU; thousands of envs need
  several GB — start smaller and scale up, see below).
- mjlab installed and importable (`cd mjlab && uv sync`), with
  `mjlab_robocup` installed as an editable package (`uv pip install -e
  sim/`) so the `Mjlab-RoboCup-Kick-v0` task id is registered (see
  [`sim/mjlab_robocup/__init__.py`](sim/mjlab_robocup/__init__.py)).
- Optional but recommended: a [Weights & Biases](https://wandb.ai)
  account for run tracking (`--logger wandb`); mjlab's `rl` package wraps
  `rsl_rl`'s `WandbLogWriter`.

### 1. Sanity-check the task before spending GPU time

Always do this first — it catches broken observation/action wiring,
NaNs, or a scene that fails to reset, without wasting a training run:

```bash
uv run play Mjlab-RoboCup-Kick-v0 --agent zero    # all-zero actions
uv run play Mjlab-RoboCup-Kick-v0 --agent random  # uniform random actions
```

Watch that: the robot and ball spawn on the field without exploding/
clipping through geometry, episodes terminate at `episode_length_s`
(20s) or on `goal_scored`/`out_of_bounds`, and the depth-scan rangefinders
visibly hit the ball/goal geometry in the viewer.

### 2. Train

```bash
Training training daily can you sort of relationships in the Flock to reset currents okay, I can golab-RoboCup-Kick-v0 --env.scene.num-envs 4096 --logger wandb
```

- **`--env.scene.num-envs`**: start at a few hundred if you're unsure your
  GPU can fit 4096 parallel envs (depth-scan rangefinests, in particular,
  add per-env cost — 64 rays each). Scale up once you've confirmed the
  task runs cleanly.
- **Multi-GPU**: `--gpu-ids "[0, 1]"` to shard `num-envs` across GPUs.
- The policy/algorithm hyperparameters (PPO clip, learning rate, recurrent
  hidden sizes, reward discounting, etc.) live in
  [`sim/mjlab_robocup/agents/rsl_rl_ppo_cfg.py`](sim/mjlab_robocup/agents/rsl_rl_ppo_cfg.py)
  — in particular `rnn_hidden_size` and `actor_hidden_dims` **must** stay
  in sync with `tools/quantize_export/network_spec.py` and
  `firmware/src/config.rs` (see `docs/NEURAL_NETWORK.md`) if you change
  them, since the deployed firmware hard-codes those dimensions.
- Checkpoints and logs are written under mjlab's run directory (printed at
  the start of training); `--experiment-name` in the PPO cfg
  (`ev3_robocup_kick`) groups runs together in wandb.

### 3. Monitor training

Track these signals (in wandb/tensorboard) specifically for this task:

- **Per-reward-term curves** (`robot_ball_approach`, `ball_goal_progress`,
  `goal_scored`, `action_rate_l2`, `wheel_energy_l2` — see
  [`sim/mjlab_robocup/robocup_env_cfg.py`](sim/mjlab_robocup/robocup_env_cfg.py)):
  if `goal_scored` stays at zero for a long time while the other terms
  improve, the robot is learning to approach the ball but not to finish;
  consider raising the `goal_scored` reward weight or shaping an
  intermediate "ball moving toward goal" bonus more strongly.
- **Episode length**: episodes ending early and often usually means
  `out_of_bounds` is triggering too easily (check field dimensions in
  `mdp/rewards.py` against the scene) rather than the policy performing
  well.
- **Action-rate penalty**: if it dominates and the robot barely moves,
  its weight (`-0.05` by default) is too high relative to the task
  rewards — lower it.
- **KL/entropy** (standard PPO diagnostics): a collapsing entropy early on
  usually means `init_noise_std` is too low or the reward is too sparse
  this early in training.

### 4. Iterate on the task, not just the network

Most of the "training difficulty" in a from-scratch RoboCup task is reward
and curriculum design, not network size. Cheap things to try before
retraining from scratch:

- Add a small dense reward for the kicker arm making contact with the
  ball (encourages discovering the kick before it's ever accidentally
  triggered).
- Add domain randomization for sensor noise/action latency early rather
  than late — a policy trained only in a noise-free sim tends to fail
  sim2real even if it plays perfectly in the viewer.
- Use `uv run play <task-id> --wandb-run-path <run>` regularly during
  training to *watch* the policy, not just read metrics — visually
  obvious failure modes (spinning in place, freezing near the ball) are
  often fixed with one reward term, not a network change.

### 5. Evaluate a trained checkpoint

```bash
uv run play Mjlab-RoboCup-Kick-v0 --wandb-run-path your-org/mjlab/run-id
```

### 6. Record a rollout for quantization calibration

The Q15 export needs a recorded sequence of real actor observations to
calibrate activation ranges (see `docs/NEURAL_NETWORK.md` → Quantization
scheme). Use the helper script, which drives a trained checkpoint through
the task and dumps its observation stream:

```bash
python -m tools.quantize_export.record_rollout \
  --task Mjlab-RoboCup-Kick-v0 \
  --wandb-run-path your-org/mjlab/run-id \
  --steps 2000 \
  --out rollout_obs.npy
```

See [`tools/quantize_export/record_rollout.py`](tools/quantize_export/record_rollout.py)
— it needs the same mjlab environment/checkpoint-loading API verification
noted in `PLAN_SUMMARY.md`, since it drives the real mjlab env rather than
the pure-Python reference model.

### 7. Quantize and export the trained weights to Q15 fixed point

```bash
python -m tools.quantize_export.cli \
  --checkpoint path/to/checkpoint.pt \
  --rollout rollout_obs.npy \
  --out firmware/src/policy_weights.rs
```

This replaces the all-zero placeholder in
[`firmware/src/policy_weights.rs`](firmware/src/policy_weights.rs) with
the real trained weights. See
[`tools/quantize_export/README.md`](tools/quantize_export/README.md)
for details of the calibration/export pipeline.

### 8. Validate before flashing hardware

Compare the quantized network's actions against the float policy on
held-out rollout data (see `PLAN_SUMMARY.md` → Phase E) — check the
action divergence stays within an acceptable tolerance before trusting it
on a physical robot.

## Deployment

1. Cross-compile the firmware for `armv5te-unknown-linux-musleabi`.
2. Copy the binary to the EV3 (statically linked, no dependencies to
   install).
3. Wire the sensors (gyro on any input port, VL53L8CX on I2C, drive +
   kicker motors on output ports) and update the port/I2C-bus constants
   in `firmware/src/main.rs` to match your build.
4. Run the benchmark once on-device to confirm real-hardware timing, then
   run `ev3-robocup`.

## Status / known gaps

This project has been built and verified end-to-end in simulation-free
form (compiles, unit-tests pass, cross-compiles, runs under ARM emulation,
meets the 50 Hz timing budget with the placeholder network) but **has not
yet been trained or run on physical hardware**. See `PLAN_SUMMARY.md` →
"Next steps for a real deployment" for what's left.
