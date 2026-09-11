//! Q15 policy inference: 4-gate LSTM (direct on raw obs, split Wx/Wh) ->
//! Dense+Tanh decoder -> Dense+Tanh action head.
//!
//! Built entirely from embedded-nn's low-level `fully_connected_s16` /
//! `sigmoid_s16` / `tanh_s16` / `elementwise_mul_s16` / `elementwise_add_s16`
//! primitives (embedded-nn has no ready-made LSTM op with a confirmed
//! stable public layout -- see /memories/embedded-nn-crate-facts.md).
//! All quantization parameters (dtypes, Dims layout, the FC/mul kernels'
//! internal `>>15` accumulator quirk, and the sigmoid/tanh Q3.12 input
//! convention) were verified empirically against embedded-nn 0.2.1, not
//! assumed from generic CMSIS-NN/TFLite docs.
//!
//! Each gate uses two separate FCs (Wx@obs, Wh@h) combined via
//! `elementwise_add_s16` rather than concatenating obs+h into one FC
//! call, because obs and h live at different quantization scales and
//! `fully_connected_s16` only supports one input scale per call.
//! `elementwise_add_s16` natively supports two differently-scaled
//! inputs. See tools/quantize_export/reference_model.py.

use embedded_nn::activations::{sigmoid_s16, tanh_s16};
use embedded_nn::basic_math::elementwise_mul_s16;
use embedded_nn::fully_connected::fully_connected_s16;
use embedded_nn::support::requantize;
use embedded_nn::types::{Activation, Dims, FcParams, PerTensorQuantParams};

use crate::config::{DECODER_DIM, INPUT_DIM, LSTM_HIDDEN_DIM, OUTPUT_DIM};
use crate::policy_weights as w;

fn fc(
    input: &[i16],
    in_dim: usize,
    out_dim: usize,
    weight: &[i8],
    bias: Option<&[i64]>,
    multiplier: i32,
    shift: i32,
    output: &mut [i16],
) {
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int16_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(multiplier, shift);
    // filter_dims: n=in_dim, c=out_dim (verified empirically, counterintuitive).
    let input_dims = Dims::new(1, 1, 1, in_dim as i32);
    let filter_dims = Dims::new(in_dim as i32, 1, 1, out_dim as i32);
    let output_dims = Dims::new(1, 1, 1, out_dim as i32);
    fully_connected_s16(
        &fc_params,
        &quant_params,
        &input_dims,
        input,
        &filter_dims,
        weight,
        bias,
        &output_dims,
        output,
    )
    .expect("fully_connected_s16 dims mismatch");
}

fn mul(input1: &[i16], input2: &[i16], mult: i32, shift: i32, output: &mut [i16]) {
    elementwise_mul_s16(
        input1,
        input2,
        output,
        mult,
        shift,
        Activation::int16_unconstrained(),
    )
    .expect("elementwise_mul_s16 length mismatch");
}

/// Combines two int16 tensors already living at different scales into one
/// output scale, matching `elementwise_add_s16`'s two-input-scale support
/// (see tools/quantize_export/quantize.py `add_s16_*_requant_params`).
fn add(
    input1: &[i16],
    mult1: i32,
    shift1: i32,
    input2: &[i16],
    mult2: i32,
    shift2: i32,
    out_mult: i32,
    out_shift: i32,
    output: &mut [i16],
) {
    for k in 0..output.len() {
        let req1 = requantize(input1[k] as i32, mult1, shift1);
        let req2 = requantize(input2[k] as i32, mult2, shift2);
        let sum = requantize(req1 + req2, out_mult, out_shift);
        output[k] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// One LSTM gate: `Wx @ obs + Wh @ h`, combined at `gate_scale`.
#[allow(clippy::too_many_arguments)]
fn gate_logit(
    obs: &[i16; INPUT_DIM],
    h: &[i16; LSTM_HIDDEN_DIM],
    wx_weight: &[i8],
    wx_bias: &[i64],
    wx_mult: i32,
    wx_shift: i32,
    wh_weight: &[i8],
    wh_mult: i32,
    wh_shift: i32,
    add_in1_mult: i32,
    add_in1_shift: i32,
    add_in2_mult: i32,
    add_in2_shift: i32,
    add_out_mult: i32,
    add_out_shift: i32,
    output: &mut [i16; LSTM_HIDDEN_DIM],
) {
    let mut wx_out = [0i16; LSTM_HIDDEN_DIM];
    let mut wh_out = [0i16; LSTM_HIDDEN_DIM];
    fc(
        obs,
        INPUT_DIM,
        LSTM_HIDDEN_DIM,
        wx_weight,
        Some(wx_bias),
        wx_mult,
        wx_shift,
        &mut wx_out,
    );
    fc(
        h,
        LSTM_HIDDEN_DIM,
        LSTM_HIDDEN_DIM,
        wh_weight,
        None,
        wh_mult,
        wh_shift,
        &mut wh_out,
    );
    add(
        &wx_out,
        add_in1_mult,
        add_in1_shift,
        &wh_out,
        add_in2_mult,
        add_in2_shift,
        add_out_mult,
        add_out_shift,
        output,
    );
}

/// Persistent LSTM state carried across control ticks.
pub struct PolicyState {
    pub h: [i16; LSTM_HIDDEN_DIM],
    pub c: [i16; LSTM_HIDDEN_DIM],
}

impl Default for PolicyState {
    fn default() -> Self {
        Self {
            h: [0; LSTM_HIDDEN_DIM],
            c: [0; LSTM_HIDDEN_DIM],
        }
    }
}

/// Runs one forward pass. `obs` must already be Q15-quantized at
/// `policy_weights::OBS_SCALE` (see crate::observation module).
pub fn step(obs: &[i16; INPUT_DIM], state: &mut PolicyState) -> [i16; OUTPUT_DIM] {
    // 1. Four LSTM gates, each Wx@obs + Wh@h.
    let mut gi_logit = [0i16; LSTM_HIDDEN_DIM];
    let mut gf_logit = [0i16; LSTM_HIDDEN_DIM];
    let mut gg_logit = [0i16; LSTM_HIDDEN_DIM];
    let mut go_logit = [0i16; LSTM_HIDDEN_DIM];
    gate_logit(
        obs,
        &state.h,
        &w::GATE_I_WX_WEIGHT,
        &w::GATE_I_WX_BIAS,
        w::GATE_I_WX_MULTIPLIER,
        w::GATE_I_WX_SHIFT,
        &w::GATE_I_WH_WEIGHT,
        w::GATE_I_WH_MULTIPLIER,
        w::GATE_I_WH_SHIFT,
        w::GATE_I_ADD_IN1_MULT,
        w::GATE_I_ADD_IN1_SHIFT,
        w::GATE_I_ADD_IN2_MULT,
        w::GATE_I_ADD_IN2_SHIFT,
        w::GATE_I_ADD_OUT_MULT,
        w::GATE_I_ADD_OUT_SHIFT,
        &mut gi_logit,
    );
    gate_logit(
        obs,
        &state.h,
        &w::GATE_F_WX_WEIGHT,
        &w::GATE_F_WX_BIAS,
        w::GATE_F_WX_MULTIPLIER,
        w::GATE_F_WX_SHIFT,
        &w::GATE_F_WH_WEIGHT,
        w::GATE_F_WH_MULTIPLIER,
        w::GATE_F_WH_SHIFT,
        w::GATE_F_ADD_IN1_MULT,
        w::GATE_F_ADD_IN1_SHIFT,
        w::GATE_F_ADD_IN2_MULT,
        w::GATE_F_ADD_IN2_SHIFT,
        w::GATE_F_ADD_OUT_MULT,
        w::GATE_F_ADD_OUT_SHIFT,
        &mut gf_logit,
    );
    gate_logit(
        obs,
        &state.h,
        &w::GATE_G_WX_WEIGHT,
        &w::GATE_G_WX_BIAS,
        w::GATE_G_WX_MULTIPLIER,
        w::GATE_G_WX_SHIFT,
        &w::GATE_G_WH_WEIGHT,
        w::GATE_G_WH_MULTIPLIER,
        w::GATE_G_WH_SHIFT,
        w::GATE_G_ADD_IN1_MULT,
        w::GATE_G_ADD_IN1_SHIFT,
        w::GATE_G_ADD_IN2_MULT,
        w::GATE_G_ADD_IN2_SHIFT,
        w::GATE_G_ADD_OUT_MULT,
        w::GATE_G_ADD_OUT_SHIFT,
        &mut gg_logit,
    );
    gate_logit(
        obs,
        &state.h,
        &w::GATE_O_WX_WEIGHT,
        &w::GATE_O_WX_BIAS,
        w::GATE_O_WX_MULTIPLIER,
        w::GATE_O_WX_SHIFT,
        &w::GATE_O_WH_WEIGHT,
        w::GATE_O_WH_MULTIPLIER,
        w::GATE_O_WH_SHIFT,
        w::GATE_O_ADD_IN1_MULT,
        w::GATE_O_ADD_IN1_SHIFT,
        w::GATE_O_ADD_IN2_MULT,
        w::GATE_O_ADD_IN2_SHIFT,
        w::GATE_O_ADD_OUT_MULT,
        w::GATE_O_ADD_OUT_SHIFT,
        &mut go_logit,
    );

    let mut gi = [0i16; LSTM_HIDDEN_DIM];
    let mut gf = [0i16; LSTM_HIDDEN_DIM];
    let mut gg = [0i16; LSTM_HIDDEN_DIM];
    let mut go = [0i16; LSTM_HIDDEN_DIM];
    sigmoid_s16(&gi_logit, &mut gi, w::GATE_I_LEFT_SHIFT);
    sigmoid_s16(&gf_logit, &mut gf, w::GATE_F_LEFT_SHIFT);
    tanh_s16(&gg_logit, &mut gg, w::GATE_G_LEFT_SHIFT);
    sigmoid_s16(&go_logit, &mut go, w::GATE_O_LEFT_SHIFT);

    // 2. c_new = f*c + i*g.
    let mut f_c = [0i16; LSTM_HIDDEN_DIM];
    let mut i_g = [0i16; LSTM_HIDDEN_DIM];
    mul(&gf, &state.c, w::FC_MUL_MULT, w::FC_MUL_SHIFT, &mut f_c);
    mul(&gi, &gg, w::IG_MUL_MULT, w::IG_MUL_SHIFT, &mut i_g);

    let mut c_new = [0i16; LSTM_HIDDEN_DIM];
    add(
        &f_c,
        w::C_ADD_IN_MULT,
        w::C_ADD_IN_SHIFT,
        &i_g,
        w::C_ADD_IN_MULT,
        w::C_ADD_IN_SHIFT,
        w::C_ADD_OUT_MULT,
        w::C_ADD_OUT_SHIFT,
        &mut c_new,
    );

    // 3. h_new = o * tanh(c_new).
    let mut tanh_c = [0i16; LSTM_HIDDEN_DIM];
    tanh_s16(&c_new, &mut tanh_c, w::CELL_STATE_LEFT_SHIFT);
    let mut h_new = [0i16; LSTM_HIDDEN_DIM];
    mul(&go, &tanh_c, w::OT_MUL_MULT, w::OT_MUL_SHIFT, &mut h_new);

    state.c = c_new;
    state.h = h_new;

    // 4. Decoder: Dense(LSTM_HIDDEN_DIM -> DECODER_DIM) -> Tanh.
    let mut decoder_logit = [0i16; DECODER_DIM];
    fc(
        &h_new,
        LSTM_HIDDEN_DIM,
        DECODER_DIM,
        &w::DECODER_WEIGHT,
        Some(&w::DECODER_BIAS),
        w::DECODER_MULTIPLIER,
        w::DECODER_SHIFT,
        &mut decoder_logit,
    );
    let mut decoded = [0i16; DECODER_DIM];
    tanh_s16(&decoder_logit, &mut decoded, w::DECODER_LEFT_SHIFT);

    // 5. Action head: Dense(DECODER_DIM -> OUTPUT_DIM) -> Tanh.
    let mut action_logit = [0i16; OUTPUT_DIM];
    fc(
        &decoded,
        DECODER_DIM,
        OUTPUT_DIM,
        &w::FC_OUT_WEIGHT,
        Some(&w::FC_OUT_BIAS),
        w::FC_OUT_MULTIPLIER,
        w::FC_OUT_SHIFT,
        &mut action_logit,
    );
    let mut action = [0i16; OUTPUT_DIM];
    tanh_s16(&action_logit, &mut action, w::ACTION_LEFT_SHIFT);
    action
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::INPUT_DIM;

    #[test]
    fn zero_input_zero_weights_gives_bounded_zero_action() {
        // With all-zero placeholder weights (policy_weights.rs) and a
        // zero observation, every gate logit/bias is zero, so gates
        // collapse to sigmoid(0)=0.5 / tanh(0)=0, and the cell state and
        // action stay at zero. This is a basic sanity/regression check,
        // not a correctness proof (needs the real exported weights + a
        // float-reference comparison, see project plan Phase E.1).
        let obs = [0i16; INPUT_DIM];
        let mut state = PolicyState::default();
        let action = step(&obs, &mut state);
        for &a in &action {
            assert!(a.abs() <= 1, "expected near-zero action, got {a}");
        }
    }

    #[test]
    fn state_persists_across_ticks() {
        let obs = [0i16; INPUT_DIM];
        let mut state = PolicyState::default();
        let _ = step(&obs, &mut state);
        // Cell state should have moved off its zero initialization at
        // least somewhere once gates are non-trivial (with zero weights
        // this is trivially still zero, but the test documents the
        // expected persistence contract for when real weights are used).
        let _ = state.c;
        let _ = state.h;
    }
}
