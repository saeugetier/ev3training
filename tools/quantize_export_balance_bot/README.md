# quantize_export_balance_bot

Converts a trained mjlab/rsl_rl balance-bot policy checkpoint into Q15
(int16) fixed-point weights, for use by embedded-nn's
`fully_connected_s16` on the EV3 -- no ONNX/TFLite step, mirroring
`tools/quantize_export_robocup` but adapted to this task's simpler
feed-forward topology (no LSTM).

## Pipeline

1. `reference_model.py` -- float reference of the exact deployed topology
   (3x `Linear + LeakyReLU` -> plain `Linear` mean head, no recurrent state
   and no tanh output since `rl_cfg.py` uses a plain `GaussianDistribution`).
   Adapt trained weights into it with `load_from_rsl_rl_state_dict`, and
   fold the `obs_normalization=True` running normalizer into `fc1` with
   `fold_input_normalization` so no normalization step is needed on-device.
2. `record_rollout.py` -- drives a trained checkpoint through the real
   mjlab task and dumps its actor observation stream as a `[T, 36]`
   float32 `.npy` file:
   ```bash
   python -m tools.quantize_export_balance_bot.record_rollout \
     --task Mjlab-Drive-BalanceBot \
     --checkpoint logs/rsl_rl/balance_bot/<run>/model_2500.pt \
     --steps 2000 \
     --out rollout_obs.npy
   ```
   `--checkpoint` is a local `.pt` file -- rsl_rl saves checkpoints locally
   regardless of the logger backend, so a tensorboard-only run works the
   same as a wandb-tracked one.
3. `calibrate.py` -- collect per-layer pre-activation absmax over the
   rollout (every intermediate tensor is a plain FC logit, no sigmoid/tanh
   gates to calibrate).
4. `export_rust.py` / `cli.py` -- quantize weights/biases, compute
   TFLite-style multiplier+shift requant params, and emit a Rust source
   file of the quantized layers plus a hand-written `leaky_relu_i16` op
   (embedded-nn has no LeakyReLU kernel).

```bash
python -m tools.quantize_export_balance_bot.cli \
  --checkpoint runs/balance_bot/model.pt \
  --rollout rollout_obs.npy \
  --out policy_weights.rs
```

Before flashing hardware, validate the quantized math against the float
reference (`BalanceBotPolicyRef`) on held-out rollout data.

## Output-to-motor mapping

The exported network's final layer produces a raw (unbounded) mean action
per wheel. On both the simulated `<motor>` actuator (`ctrlrange="-1 1"`)
and the real EV3 (`set_duty_cycle_sp`, open-loop duty cycle, no torque or
speed servo), this value must be clamped to `[-1, 1]` and then mapped
linearly to a `-100..100` duty cycle -- see
`balance_robot_rl/README.md` and `firmware/src/sensors/tacho.rs` for the
rationale (the policy was trained against a torque/effort actuator to
match the EV3's lack of closed-loop torque or speed control).
