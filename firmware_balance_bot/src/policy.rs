//! Q15 policy inference: plain feed-forward MLP (no recurrent state),
//! matching rsl_rl's stock `ActorCritic` actor -- 3x `Linear + LeakyReLU`
//! hidden layers, then a plain `Linear` mean head (no tanh, since
//! `rl_cfg.py`'s `distribution_cfg` is a plain `GaussianDistribution`; the
//! actuator/duty-cycle clamp bounds the action instead of a squashing
//! activation -- see `actuation::dequantize_effort`).
//!
//! Uses `embedded_nn::fully_connected::fully_connected_s16` for each
//! layer, same as the RoboCup kick firmware. embedded-nn has no LeakyReLU
//! kernel, so it's applied via `policy_weights::leaky_relu_i16`, a small
//! hand-written int16 op (see `tools/quantize_export_balance_bot/export_rust.py`).

use embedded_nn::fully_connected::fully_connected_s16;
use embedded_nn::types::{Activation, Dims, FcParams, PerTensorQuantParams};

use crate::config::{HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM, INPUT_DIM, OUTPUT_DIM};
use crate::policy_weights as w;

pub struct PolicyTrace {
    pub fc1: [i16; HIDDEN1_DIM],
    pub fc2: [i16; HIDDEN2_DIM],
    pub fc3: [i16; HIDDEN3_DIM],
    pub action: [i16; OUTPUT_DIM],
}

fn fc(
    input: &[i16],
    in_dim: usize,
    out_dim: usize,
    weight: &[i8],
    bias: &[i64],
    multiplier: &[i32],
    shift: &[i32],
    output: &mut [i16],
) {
    // Per-output-channel (per-row) quantization: a single output neuron
    // per `fully_connected_s16` call, each with its own weight scale, so
    // one large-magnitude outlier weight doesn't coarsen every other
    // neuron's int8 resolution (see export_rust.py's `_export_layer`).
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int16_unconstrained(),
    };
    let input_dims = Dims::new(1, 1, 1, in_dim as i32);
    let filter_dims = Dims::new(in_dim as i32, 1, 1, 1);
    let output_dims = Dims::new(1, 1, 1, 1);
    for row in 0..out_dim {
        let quant_params = PerTensorQuantParams::new(multiplier[row], shift[row]);
        let weight_row = &weight[row * in_dim..(row + 1) * in_dim];
        let bias_row = &bias[row..row + 1];
        fully_connected_s16(
            &fc_params,
            &quant_params,
            &input_dims,
            input,
            &filter_dims,
            weight_row,
            Some(bias_row),
            &output_dims,
            &mut output[row..row + 1],
        )
        .expect("fully_connected_s16 dims mismatch");
    }
}

/// Runs one forward pass. `obs` must already be Q15-quantized at
/// `policy_weights::OBS_SCALE` (see crate::observation module).
pub fn infer(obs: &[i16; INPUT_DIM]) -> [i16; OUTPUT_DIM] {
    infer_trace(obs).action
}

pub fn infer_trace(obs: &[i16; INPUT_DIM]) -> PolicyTrace {
    let mut h1 = [0i16; HIDDEN1_DIM];
    fc(
        obs,
        INPUT_DIM,
        HIDDEN1_DIM,
        &w::FC1_WEIGHT,
        &w::FC1_BIAS,
        &w::FC1_MULTIPLIER,
        &w::FC1_SHIFT,
        &mut h1,
    );
    for v in h1.iter_mut() {
        *v = w::leaky_relu_i16(*v);
    }

    let mut h2 = [0i16; HIDDEN2_DIM];
    fc(
        &h1,
        HIDDEN1_DIM,
        HIDDEN2_DIM,
        &w::FC2_WEIGHT,
        &w::FC2_BIAS,
        &w::FC2_MULTIPLIER,
        &w::FC2_SHIFT,
        &mut h2,
    );
    for v in h2.iter_mut() {
        *v = w::leaky_relu_i16(*v);
    }

    let mut h3 = [0i16; HIDDEN3_DIM];
    fc(
        &h2,
        HIDDEN2_DIM,
        HIDDEN3_DIM,
        &w::FC3_WEIGHT,
        &w::FC3_BIAS,
        &w::FC3_MULTIPLIER,
        &w::FC3_SHIFT,
        &mut h3,
    );
    for v in h3.iter_mut() {
        *v = w::leaky_relu_i16(*v);
    }

    let mut action = [0i16; OUTPUT_DIM];
    fc(
        &h3,
        HIDDEN3_DIM,
        OUTPUT_DIM,
        &w::FC_OUT_WEIGHT,
        &w::FC_OUT_BIAS,
        &w::FC_OUT_MULTIPLIER,
        &w::FC_OUT_SHIFT,
        &mut action,
    );
    PolicyTrace {
        fc1: h1,
        fc2: h2,
        fc3: h3,
        action,
    }
}
