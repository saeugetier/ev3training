"""Export a calibrated `RoboCupPolicyRef` to a Rust source file of
int8-weight / int16-activation Q-format arrays plus per-layer
requantization params, for direct use by firmware/src/policy.rs via
`embedded_nn::fully_connected::fully_connected_s16` (no ONNX/TFLite
intermediate -- embedded-nn has no LSTM importer, see project plan).

Weight/bias dtypes and the multiplier formula follow embedded-nn's actual
verified behavior, not generic TFLite conventions -- see
/memories/embedded-nn-crate-facts.md.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from tools.quantize_export.calibrate import FIXED_UNIT_SCALE, CalibrationStats, GateStats
from tools.quantize_export.quantize import (
    add_s16_input_requant_params,
    add_s16_output_requant_params,
    calibrate_symmetric_scale,
    fc_s16_requant_params,
    mul_s16_requant_params,
    pow2_activation_scale,
    quantize_bias_s64,
    quantize_weight_s8,
)


def _fmt_i8_array(name: str, values: np.ndarray) -> str:
    flat = ", ".join(str(int(v)) for v in values.flatten())
    return f"pub static {name}: [i8; {values.size}] = [{flat}];\n"


def _fmt_i64_array(name: str, values: np.ndarray) -> str:
    flat = ", ".join(str(int(v)) for v in values.flatten())
    return f"pub static {name}: [i64; {values.size}] = [{flat}];\n"


def _export_layer(
    prefix: str,
    layer: nn.Linear,
    input_scale: float,
    output_scale: float,
    with_bias: bool = True,
) -> str:
    weight = layer.weight.detach().numpy()  # [out, in], row-major.
    weight_scale = calibrate_symmetric_scale(weight, num_bits=8)
    q_weight = quantize_weight_s8(weight, weight_scale)
    multiplier, shift = fc_s16_requant_params(input_scale, weight_scale, output_scale)
    out_dim, in_dim = weight.shape

    lines = [
        f"// {prefix}: Linear({in_dim} -> {out_dim}), input_scale={input_scale:.8g}, "
        f"weight_scale={weight_scale:.8g}, output_scale={output_scale:.8g}",
        _fmt_i8_array(f"{prefix}_WEIGHT", q_weight),
    ]
    if with_bias:
        bias = layer.bias.detach().numpy()
        q_bias = quantize_bias_s64(bias, input_scale, weight_scale)
        lines.append(_fmt_i64_array(f"{prefix}_BIAS", q_bias))
    lines += [
        f"pub const {prefix}_IN_DIM: usize = {in_dim};",
        f"pub const {prefix}_OUT_DIM: usize = {out_dim};",
        f"pub const {prefix}_MULTIPLIER: i32 = {multiplier};",
        f"pub const {prefix}_SHIFT: i32 = {shift};",
        "",
    ]
    return "\n".join(lines)


def _export_gate(
    name: str,
    gate_x: nn.Linear,
    gate_h: nn.Linear,
    obs_scale: float,
    stats: GateStats,
) -> tuple[str, float, int]:
    """Export one gate's Wx/Wh FCs plus the elementwise_add_s16 params that
    combine them, and its sigmoid_s16/tanh_s16 left_shift. Returns
    (rust_source, gate_scale, gate_left_shift)."""
    wx_scale = max(stats.wx_absmax / 32767.0, 1e-8)
    wh_scale = max(stats.wh_absmax / 32767.0, 1e-8)
    gate_scale, gate_left_shift = pow2_activation_scale(stats.logit_absmax)

    add_in1_mult, add_in1_shift = add_s16_input_requant_params(wx_scale, gate_scale)
    add_in2_mult, add_in2_shift = add_s16_input_requant_params(wh_scale, gate_scale)
    add_out_mult, add_out_shift = add_s16_output_requant_params(gate_scale, gate_scale)

    src = "\n".join(
        [
            _export_layer(f"{name}_WX", gate_x, obs_scale, wx_scale),
            _export_layer(f"{name}_WH", gate_h, FIXED_UNIT_SCALE, wh_scale, with_bias=False),
            f"pub const {name}_SCALE: f32 = {gate_scale:.8g};",
            f"pub const {name}_LEFT_SHIFT: i32 = {gate_left_shift};",
            f"pub const {name}_ADD_IN1_MULT: i32 = {add_in1_mult};",
            f"pub const {name}_ADD_IN1_SHIFT: i32 = {add_in1_shift};",
            f"pub const {name}_ADD_IN2_MULT: i32 = {add_in2_mult};",
            f"pub const {name}_ADD_IN2_SHIFT: i32 = {add_in2_shift};",
            f"pub const {name}_ADD_OUT_MULT: i32 = {add_out_mult};",
            f"pub const {name}_ADD_OUT_SHIFT: i32 = {add_out_shift};",
            "",
        ]
    )
    return src, gate_scale, gate_left_shift


def export_policy_weights_rs(model, stats: CalibrationStats) -> str:
    obs_absmax = float(np.max(np.maximum(np.abs(stats.obs_min), np.abs(stats.obs_max))))
    obs_scale = max(obs_absmax / 32767.0, 1e-8)

    # Cell state feeds tanh_s16 (h = o * tanh(c)), so it needs a
    # power-of-two Q3.12-compatible scale (see quantize.py).
    cell_state_scale, cell_state_left_shift = pow2_activation_scale(stats.cell_state_absmax)
    decoder_out_scale, decoder_left_shift = pow2_activation_scale(stats.decoder_logit_absmax)
    action_logit_scale, action_left_shift = pow2_activation_scale(stats.action_logit_absmax)

    gate_i_src, _, _ = _export_gate("GATE_I", model.gate_i_x, model.gate_i_h, obs_scale, stats.gate_i)
    gate_f_src, _, _ = _export_gate("GATE_F", model.gate_f_x, model.gate_f_h, obs_scale, stats.gate_f)
    gate_g_src, _, _ = _export_gate("GATE_G", model.gate_g_x, model.gate_g_h, obs_scale, stats.gate_g)
    gate_o_src, _, _ = _export_gate("GATE_O", model.gate_o_x, model.gate_o_h, obs_scale, stats.gate_o)

    # LSTM gate-combination requant params: c_new = f*c + i*g; h_new = o*tanh(c_new).
    fc_mult, fc_shift = mul_s16_requant_params(FIXED_UNIT_SCALE, cell_state_scale, cell_state_scale)
    ig_mult, ig_shift = mul_s16_requant_params(FIXED_UNIT_SCALE, FIXED_UNIT_SCALE, cell_state_scale)
    add_in_mult, add_in_shift = add_s16_input_requant_params(cell_state_scale, cell_state_scale)
    add_out_mult, add_out_shift = add_s16_output_requant_params(cell_state_scale, cell_state_scale)
    ot_mult, ot_shift = mul_s16_requant_params(FIXED_UNIT_SCALE, FIXED_UNIT_SCALE, FIXED_UNIT_SCALE)

    chunks = [
        "// AUTO-GENERATED by tools/quantize_export/export_rust.py. Do not edit by hand.",
        "#![allow(clippy::all)]",
        "",
        f"pub const OBS_SCALE: f32 = {obs_scale:.8g};",
        f"pub const CELL_STATE_SCALE: f32 = {cell_state_scale:.8g};",
        f"pub const CELL_STATE_LEFT_SHIFT: i32 = {cell_state_left_shift};  // for tanh_s16(c)",
        f"pub const DECODER_OUT_SCALE: f32 = {decoder_out_scale:.8g};",
        f"pub const DECODER_LEFT_SHIFT: i32 = {decoder_left_shift};  // for tanh_s16",
        f"pub const ACTION_LOGIT_SCALE: f32 = {action_logit_scale:.8g};",
        f"pub const ACTION_LEFT_SHIFT: i32 = {action_left_shift};  // for tanh_s16",
        f"pub const UNIT_SCALE: f32 = {FIXED_UNIT_SCALE:.8g};  // sigmoid/tanh Q0.15 output (raw/32768)",
        "",
        "// LSTM gate-combination requant params: c_new = f*c + i*g; h_new = o*tanh(c_new).",
        f"pub const FC_MUL_MULT: i32 = {fc_mult};",
        f"pub const FC_MUL_SHIFT: i32 = {fc_shift};",
        f"pub const IG_MUL_MULT: i32 = {ig_mult};",
        f"pub const IG_MUL_SHIFT: i32 = {ig_shift};",
        f"pub const C_ADD_IN_MULT: i32 = {add_in_mult};",
        f"pub const C_ADD_IN_SHIFT: i32 = {add_in_shift};",
        f"pub const C_ADD_OUT_MULT: i32 = {add_out_mult};",
        f"pub const C_ADD_OUT_SHIFT: i32 = {add_out_shift};",
        f"pub const OT_MUL_MULT: i32 = {ot_mult};",
        f"pub const OT_MUL_SHIFT: i32 = {ot_shift};",
        "",
        gate_i_src,
        gate_f_src,
        gate_g_src,
        gate_o_src,
        _export_layer("DECODER", model.decoder, FIXED_UNIT_SCALE, decoder_out_scale),
        _export_layer("FC_OUT", model.fc_out, FIXED_UNIT_SCALE, action_logit_scale),
    ]
    return "\n".join(chunks)
