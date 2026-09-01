//! PLACEHOLDER weights (all zero) so the firmware crate compiles before
//! `tools/quantize_export/cli.py` has been run against a real trained
//! checkpoint. Replace this entire file with the generated output before
//! flashing hardware -- see tools/quantize_export/README.md.
#![allow(clippy::all)]

use crate::config::{DECODER_DIM, INPUT_DIM, LSTM_HIDDEN_DIM, OUTPUT_DIM};

pub const OBS_SCALE: f32 = 1.0;
pub const CELL_STATE_SCALE: f32 = 1.0 / 4096.0;
pub const CELL_STATE_LEFT_SHIFT: i32 = 0;
pub const DECODER_OUT_SCALE: f32 = 1.0 / 4096.0;
pub const DECODER_LEFT_SHIFT: i32 = 0;
pub const ACTION_LOGIT_SCALE: f32 = 1.0 / 4096.0;
pub const ACTION_LEFT_SHIFT: i32 = 0;
pub const UNIT_SCALE: f32 = 1.0 / 32768.0;

// Identity-ish placeholders (multiplier for real_multiplier=1.0, Q31 conv.).
const IDENT_MULT: i32 = 1 << 30;
const IDENT_SHIFT: i32 = 1;

pub const FC_MUL_MULT: i32 = IDENT_MULT;
pub const FC_MUL_SHIFT: i32 = IDENT_SHIFT;
pub const IG_MUL_MULT: i32 = IDENT_MULT;
pub const IG_MUL_SHIFT: i32 = IDENT_SHIFT;
pub const C_ADD_IN_MULT: i32 = IDENT_MULT;
pub const C_ADD_IN_SHIFT: i32 = IDENT_SHIFT;
pub const C_ADD_OUT_MULT: i32 = IDENT_MULT;
pub const C_ADD_OUT_SHIFT: i32 = IDENT_SHIFT;
pub const OT_MUL_MULT: i32 = IDENT_MULT;
pub const OT_MUL_SHIFT: i32 = IDENT_SHIFT;

pub const GATE_I_WX_WEIGHT: [i8; INPUT_DIM * LSTM_HIDDEN_DIM] = [0; INPUT_DIM * LSTM_HIDDEN_DIM];
pub const GATE_I_WX_BIAS: [i64; LSTM_HIDDEN_DIM] = [0; LSTM_HIDDEN_DIM];
pub const GATE_I_WX_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_I_WX_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_I_WH_WEIGHT: [i8; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM] =
    [0; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM];
pub const GATE_I_WH_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_I_WH_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_I_SCALE: f32 = 1.0 / 4096.0;
pub const GATE_I_LEFT_SHIFT: i32 = 0;
pub const GATE_I_ADD_IN1_MULT: i32 = IDENT_MULT;
pub const GATE_I_ADD_IN1_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_I_ADD_IN2_MULT: i32 = IDENT_MULT;
pub const GATE_I_ADD_IN2_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_I_ADD_OUT_MULT: i32 = IDENT_MULT;
pub const GATE_I_ADD_OUT_SHIFT: i32 = IDENT_SHIFT;

pub const GATE_F_WX_WEIGHT: [i8; INPUT_DIM * LSTM_HIDDEN_DIM] = [0; INPUT_DIM * LSTM_HIDDEN_DIM];
pub const GATE_F_WX_BIAS: [i64; LSTM_HIDDEN_DIM] = [0; LSTM_HIDDEN_DIM];
pub const GATE_F_WX_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_F_WX_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_F_WH_WEIGHT: [i8; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM] =
    [0; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM];
pub const GATE_F_WH_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_F_WH_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_F_SCALE: f32 = 1.0 / 4096.0;
pub const GATE_F_LEFT_SHIFT: i32 = 0;
pub const GATE_F_ADD_IN1_MULT: i32 = IDENT_MULT;
pub const GATE_F_ADD_IN1_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_F_ADD_IN2_MULT: i32 = IDENT_MULT;
pub const GATE_F_ADD_IN2_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_F_ADD_OUT_MULT: i32 = IDENT_MULT;
pub const GATE_F_ADD_OUT_SHIFT: i32 = IDENT_SHIFT;

pub const GATE_G_WX_WEIGHT: [i8; INPUT_DIM * LSTM_HIDDEN_DIM] = [0; INPUT_DIM * LSTM_HIDDEN_DIM];
pub const GATE_G_WX_BIAS: [i64; LSTM_HIDDEN_DIM] = [0; LSTM_HIDDEN_DIM];
pub const GATE_G_WX_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_G_WX_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_G_WH_WEIGHT: [i8; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM] =
    [0; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM];
pub const GATE_G_WH_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_G_WH_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_G_SCALE: f32 = 1.0 / 4096.0;
pub const GATE_G_LEFT_SHIFT: i32 = 0;
pub const GATE_G_ADD_IN1_MULT: i32 = IDENT_MULT;
pub const GATE_G_ADD_IN1_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_G_ADD_IN2_MULT: i32 = IDENT_MULT;
pub const GATE_G_ADD_IN2_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_G_ADD_OUT_MULT: i32 = IDENT_MULT;
pub const GATE_G_ADD_OUT_SHIFT: i32 = IDENT_SHIFT;

pub const GATE_O_WX_WEIGHT: [i8; INPUT_DIM * LSTM_HIDDEN_DIM] = [0; INPUT_DIM * LSTM_HIDDEN_DIM];
pub const GATE_O_WX_BIAS: [i64; LSTM_HIDDEN_DIM] = [0; LSTM_HIDDEN_DIM];
pub const GATE_O_WX_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_O_WX_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_O_WH_WEIGHT: [i8; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM] =
    [0; LSTM_HIDDEN_DIM * LSTM_HIDDEN_DIM];
pub const GATE_O_WH_MULTIPLIER: i32 = IDENT_MULT;
pub const GATE_O_WH_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_O_SCALE: f32 = 1.0 / 4096.0;
pub const GATE_O_LEFT_SHIFT: i32 = 0;
pub const GATE_O_ADD_IN1_MULT: i32 = IDENT_MULT;
pub const GATE_O_ADD_IN1_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_O_ADD_IN2_MULT: i32 = IDENT_MULT;
pub const GATE_O_ADD_IN2_SHIFT: i32 = IDENT_SHIFT;
pub const GATE_O_ADD_OUT_MULT: i32 = IDENT_MULT;
pub const GATE_O_ADD_OUT_SHIFT: i32 = IDENT_SHIFT;

pub const DECODER_WEIGHT: [i8; LSTM_HIDDEN_DIM * DECODER_DIM] = [0; LSTM_HIDDEN_DIM * DECODER_DIM];
pub const DECODER_BIAS: [i64; DECODER_DIM] = [0; DECODER_DIM];
pub const DECODER_MULTIPLIER: i32 = IDENT_MULT;
pub const DECODER_SHIFT: i32 = IDENT_SHIFT;

pub const FC_OUT_WEIGHT: [i8; DECODER_DIM * OUTPUT_DIM] = [0; DECODER_DIM * OUTPUT_DIM];
pub const FC_OUT_BIAS: [i64; OUTPUT_DIM] = [0; OUTPUT_DIM];
pub const FC_OUT_MULTIPLIER: i32 = IDENT_MULT;
pub const FC_OUT_SHIFT: i32 = IDENT_SHIFT;
