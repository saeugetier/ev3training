"""Activation range calibration for Q15 export.

Most intermediate activations in the policy are inherently bounded to
[-1, 1] by construction (tanh/sigmoid outputs), so they use a fixed
scale. What genuinely needs data-driven calibration are: the raw
observation input, each gate's two partial pre-activation sums (Wx@obs
and Wh@h, computed/requantized separately -- see reference_model.py),
each gate's combined pre-activation logit, the cell state `c`, the
decoder's pre-tanh logit, and the final action pre-tanh logit. Run this
over a recorded rollout (ideally from `uv run play` on the trained
checkpoint) before exporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from tools.quantize_export.reference_model import RoboCupPolicyRef

# Activations bounded by tanh/sigmoid: fixed Q15 scale, no calibration needed.
FIXED_UNIT_SCALE = 1.0 / 32767.0


@dataclass
class GateStats:
    wx_absmax: float = 0.0
    wh_absmax: float = 0.0
    logit_absmax: float = 0.0  # combined Wx@obs + Wh@h, pre-activation.


@dataclass
class CalibrationStats:
    obs_min: np.ndarray = field(default_factory=lambda: None)
    obs_max: np.ndarray = field(default_factory=lambda: None)
    gate_i: GateStats = field(default_factory=GateStats)
    gate_f: GateStats = field(default_factory=GateStats)
    gate_g: GateStats = field(default_factory=GateStats)
    gate_o: GateStats = field(default_factory=GateStats)
    cell_state_absmax: float = 0.0
    decoder_logit_absmax: float = 0.0
    action_logit_absmax: float = 0.0

    def update_obs(self, obs: np.ndarray) -> None:
        if self.obs_min is None:
            self.obs_min = obs.copy()
            self.obs_max = obs.copy()
        else:
            self.obs_min = np.minimum(self.obs_min, obs)
            self.obs_max = np.maximum(self.obs_max, obs)


def _update_gate(stats: GateStats, wx: torch.Tensor, wh: torch.Tensor) -> None:
    stats.wx_absmax = max(stats.wx_absmax, wx.abs().max().item())
    stats.wh_absmax = max(stats.wh_absmax, wh.abs().max().item())
    stats.logit_absmax = max(stats.logit_absmax, (wx + wh).abs().max().item())


@torch.no_grad()
def calibrate(model: RoboCupPolicyRef, obs_sequence: np.ndarray) -> CalibrationStats:
    """`obs_sequence`: [T, INPUT_DIM] float32 array from a recorded rollout."""
    stats = CalibrationStats()
    h, c = model.init_hidden(batch_size=1)
    for t in range(obs_sequence.shape[0]):
        obs_t = torch.from_numpy(obs_sequence[t : t + 1]).float()
        stats.update_obs(obs_sequence[t])

        wx_i, wh_i = model.gate_i_x(obs_t), model.gate_i_h(h)
        wx_f, wh_f = model.gate_f_x(obs_t), model.gate_f_h(h)
        wx_g, wh_g = model.gate_g_x(obs_t), model.gate_g_h(h)
        wx_o, wh_o = model.gate_o_x(obs_t), model.gate_o_h(h)
        _update_gate(stats.gate_i, wx_i, wh_i)
        _update_gate(stats.gate_f, wx_f, wh_f)
        _update_gate(stats.gate_g, wx_g, wh_g)
        _update_gate(stats.gate_o, wx_o, wh_o)

        i = torch.sigmoid(wx_i + wh_i)
        f = torch.sigmoid(wx_f + wh_f)
        g = torch.tanh(wx_g + wh_g)
        o = torch.sigmoid(wx_o + wh_o)
        c = f * c + i * g
        stats.cell_state_absmax = max(stats.cell_state_absmax, c.abs().max().item())
        h = o * torch.tanh(c)

        decoder_logit = model.decoder(h)
        stats.decoder_logit_absmax = max(
            stats.decoder_logit_absmax, decoder_logit.abs().max().item()
        )
        decoded = torch.tanh(decoder_logit)

        action_logit = model.fc_out(decoded)
        stats.action_logit_absmax = max(
            stats.action_logit_absmax, action_logit.abs().max().item()
        )

    return stats
