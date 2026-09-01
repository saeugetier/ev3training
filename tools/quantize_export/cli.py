"""CLI: checkpoint + rollout -> calibrated Q15 Rust weight file.

Usage:
    python -m tools.quantize_export.cli \
        --checkpoint path/to/rsl_rl_checkpoint.pt \
        --rollout path/to/rollout_obs.npy \
        --out firmware/src/policy_weights.rs

`rollout_obs.npy` is a [T, INPUT_DIM] float32 array of actor observations
recorded from `uv run play` on the trained policy (see project plan
Phase C). `checkpoint` must be adapted to this repo's split-gate
`RoboCupPolicyRef` via `reference_model.load_from_rsl_rl_lstm_cell` --
see that function's docstring for the checkpoint-key assumptions to verify.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tools.quantize_export.calibrate import calibrate
from tools.quantize_export.export_rust import export_policy_weights_rs
from tools.quantize_export.reference_model import RoboCupPolicyRef


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = RoboCupPolicyRef()
    state = torch.load(args.checkpoint, map_location="cpu")
    # NOTE: adjust this key path once the real rsl_rl checkpoint layout is
    # known -- typically state["model_state_dict"] for rsl_rl's OnPolicyRunner.
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    obs_sequence = np.load(args.rollout).astype(np.float32)
    stats = calibrate(model, obs_sequence)

    rust_src = export_policy_weights_rs(model, stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rust_src)
    print(f"Wrote {args.out} ({len(rust_src)} bytes)")


if __name__ == "__main__":
    main()
