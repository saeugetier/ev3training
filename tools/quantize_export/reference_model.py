"""Pure PyTorch reference implementation of the deployed policy topology.

This mirrors exactly what the Rust firmware computes (a 4-gate LSTM
operating directly on raw observations, using split Wx/Wh weight
matrices per gate, followed by a Dense+Tanh decoder and a Dense+Tanh
action head), so it can be used both to (a) adapt/calibrate from an
rsl_rl `ActorCriticRecurrent` checkpoint and (b) validate the
Q15-quantized firmware math against float ground truth (Phase C / E of
the project plan) before flashing hardware.

The LSTM is implemented gate-by-gate with separate input-to-hidden (Wx)
and hidden-to-hidden (Wh) weights per gate -- matching PyTorch's
`nn.LSTMCell` internal layout 1:1 (no concatenation needed) -- because
embedded-nn's `fully_connected_s16` requires one fixed input scale per
call, and `obs` and the LSTM hidden state `h` live at different scales.
Combining `Wx @ x` and `Wh @ h` is done with `elementwise_add_s16` on
the firmware side, which natively supports two differently-scaled
inputs (see firmware/src/policy.rs).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tools.quantize_export.network_spec import DECODER_DIM, INPUT_DIM, LSTM_HIDDEN_DIM, OUTPUT_DIM


class RoboCupPolicyRef(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Wx: input_dim -> hidden (carries the gate bias). Wh: hidden ->
        # hidden (bias fixed at zero -- the real LSTM bias is on Wx only).
        self.gate_i_x = nn.Linear(INPUT_DIM, LSTM_HIDDEN_DIM)
        self.gate_f_x = nn.Linear(INPUT_DIM, LSTM_HIDDEN_DIM)
        self.gate_g_x = nn.Linear(INPUT_DIM, LSTM_HIDDEN_DIM)
        self.gate_o_x = nn.Linear(INPUT_DIM, LSTM_HIDDEN_DIM)
        self.gate_i_h = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, bias=False)
        self.gate_f_h = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, bias=False)
        self.gate_g_h = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, bias=False)
        self.gate_o_h = nn.Linear(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM, bias=False)

        self.decoder = nn.Linear(LSTM_HIDDEN_DIM, DECODER_DIM)
        self.fc_out = nn.Linear(DECODER_DIM, OUTPUT_DIM)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, c = hidden
        i = torch.sigmoid(self.gate_i_x(obs) + self.gate_i_h(h))
        f = torch.sigmoid(self.gate_f_x(obs) + self.gate_f_h(h))
        g = torch.tanh(self.gate_g_x(obs) + self.gate_g_h(h))
        o = torch.sigmoid(self.gate_o_x(obs) + self.gate_o_h(h))
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        decoded = torch.tanh(self.decoder(h_new))
        action = torch.tanh(self.fc_out(decoded))
        return action, (h_new, c_new)

    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, LSTM_HIDDEN_DIM)
        c = torch.zeros(batch_size, LSTM_HIDDEN_DIM)
        return h, c


def load_from_rsl_rl_lstm_cell(model: RoboCupPolicyRef, lstm_cell: nn.LSTMCell) -> None:
    """Copy weights from a trained `nn.LSTMCell` into the 8 split gate Linears.

    PyTorch's `LSTMCell` already stores `weight_ih`/`weight_hh` as the 4
    gates stacked along dim 0 in (input, forget, cell, output) order --
    see the `torch.nn.LSTMCell` docs -- so this is a direct slice, no
    concatenation. If your rsl_rl checkpoint's recurrent layer differs
    (e.g. GRU, or gate order), adjust the chunking below.
    """
    hidden = LSTM_HIDDEN_DIM
    w_ih = lstm_cell.weight_ih.detach()  # [4*hidden, INPUT_DIM]
    w_hh = lstm_cell.weight_hh.detach()  # [4*hidden, hidden]
    b_ih = lstm_cell.bias_ih.detach()
    b_hh = lstm_cell.bias_hh.detach()

    gates_x = [model.gate_i_x, model.gate_f_x, model.gate_g_x, model.gate_o_x]
    gates_h = [model.gate_i_h, model.gate_f_h, model.gate_g_h, model.gate_o_h]
    for idx, (gate_x, gate_h) in enumerate(zip(gates_x, gates_h)):
        gate_x.weight.data = w_ih[idx * hidden : (idx + 1) * hidden]
        gate_x.bias.data = b_ih[idx * hidden : (idx + 1) * hidden] + b_hh[
            idx * hidden : (idx + 1) * hidden
        ]
        gate_h.weight.data = w_hh[idx * hidden : (idx + 1) * hidden]
