# quantize_export

Converts a trained mjlab/rsl_rl RoboCup policy checkpoint into Q15
(int16) fixed-point weights consumed directly by the Rust firmware
(`firmware/src/policy.rs`) -- no ONNX/TFLite step, since embedded-nn has
no LSTM importer.

## Pipeline

1. `reference_model.py` -- float reference of the exact deployed topology
   (4-gate LSTM directly on raw obs, split Wx/Wh -> Dense+Tanh decoder ->
   Dense+Tanh action head, see `docs/NEURAL_NETWORK.md`). Adapt trained
   weights into it with `load_from_rsl_rl_lstm_cell`.
2. `record_rollout.py` -- drives a trained checkpoint through the real
   mjlab task and dumps its actor observation stream as a `[T, 72]`
   float32 `.npy` file:
   ```bash
   python -m tools.quantize_export.record_rollout \
     --task Mjlab-RoboCup-Kick-v0 \
     --wandb-run-path your-org/mjlab/run-id \
     --steps 2000 \
     --out rollout_obs.npy
   ```
3. `calibrate.py` -- collect per-tensor activation ranges over the rollout.
4. `export_rust.py` / `cli.py` -- quantize weights/biases, compute
   TFLite-style multiplier+shift requant params, emit
   `firmware/src/policy_weights.rs`.

```bash
python -m tools.quantize_export.cli \
  --checkpoint runs/robocup_kick/model.pt \
  --rollout rollout_obs.npy \
  --out ../firmware/src/policy_weights.rs
```

Before flashing hardware, validate the quantized math against the float
reference on held-out rollout data (see project plan Phase C.5 / E.1).
