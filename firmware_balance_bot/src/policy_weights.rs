//! PLACEHOLDER weights (all zero) so the firmware crate compiles before
//! `tools/quantize_export_balance_bot/cli.py` has been run against a real
//! trained checkpoint. Replace this entire file with the generated output
//! before flashing hardware -- see
//! `tools/quantize_export_balance_bot/README.md`.
#![allow(clippy::all)]

use crate::config::{HIDDEN1_DIM, HIDDEN2_DIM, HIDDEN3_DIM, INPUT_DIM, OUTPUT_DIM};

pub const OBS_SCALE: f32 = 1.0;
pub const FC1_OUT_SCALE: f32 = 1.0;
pub const FC2_OUT_SCALE: f32 = 1.0;
pub const FC3_OUT_SCALE: f32 = 1.0;
pub const ACTION_SCALE: f32 = 1.0 / 32767.0;
pub const NEG_SLOPE_Q15: i32 = 328; // round(0.01 * 32768).

/// `x >= 0 ? x : (x * NEG_SLOPE_Q15) >> 15`, same scale in and out.
pub fn leaky_relu_i16(x: i16) -> i16 {
    if x >= 0 {
        x
    } else {
        ((x as i32 * NEG_SLOPE_Q15) >> 15) as i16
    }
}

// Identity-ish placeholder (multiplier for real_multiplier=1.0, Q31 conv.).
const IDENT_MULT: i32 = 1 << 30;
const IDENT_SHIFT: i32 = 1;

pub const FC1_WEIGHT: [i8; INPUT_DIM * HIDDEN1_DIM] = [0; INPUT_DIM * HIDDEN1_DIM];
pub const FC1_BIAS: [i64; HIDDEN1_DIM] = [0; HIDDEN1_DIM];
pub const FC1_MULTIPLIER: i32 = IDENT_MULT;
pub const FC1_SHIFT: i32 = IDENT_SHIFT;

pub const FC2_WEIGHT: [i8; HIDDEN1_DIM * HIDDEN2_DIM] = [0; HIDDEN1_DIM * HIDDEN2_DIM];
pub const FC2_BIAS: [i64; HIDDEN2_DIM] = [0; HIDDEN2_DIM];
pub const FC2_MULTIPLIER: i32 = IDENT_MULT;
pub const FC2_SHIFT: i32 = IDENT_SHIFT;

pub const FC3_WEIGHT: [i8; HIDDEN2_DIM * HIDDEN3_DIM] = [0; HIDDEN2_DIM * HIDDEN3_DIM];
pub const FC3_BIAS: [i64; HIDDEN3_DIM] = [0; HIDDEN3_DIM];
pub const FC3_MULTIPLIER: i32 = IDENT_MULT;
pub const FC3_SHIFT: i32 = IDENT_SHIFT;

pub const FC_OUT_WEIGHT: [i8; HIDDEN3_DIM * OUTPUT_DIM] = [0; HIDDEN3_DIM * OUTPUT_DIM];
pub const FC_OUT_BIAS: [i64; OUTPUT_DIM] = [0; OUTPUT_DIM];
pub const FC_OUT_MULTIPLIER: i32 = IDENT_MULT;
pub const FC_OUT_SHIFT: i32 = IDENT_SHIFT;
