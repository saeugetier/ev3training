from __future__ import annotations

import argparse
import numpy as np
import tensorflow as tf

from tools.quantize_export_balance_bot.cli import _extract_state_dict
from tools.quantize_export_balance_bot.reference_model import BalanceBotPolicyRef
from tools.quantize_export_balance_bot.network_spec import INPUT_DIM

def load_tflite_model(model_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_input_output_details(interpreter: tf.lite.Interpreter) -> tuple[list[int], list[int]]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def validate_model(interpreter: tf.lite.Interpreter, input_data: np.ndarray, expected_output: np.ndarray) -> None:
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    
    if not np.allclose(output_data, expected_output, atol=1e-5):
        raise ValueError("Output does not match expected output.")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite-model", type=str, required=True, help="Path to the TFLite model.")
    parser.add_argument("--expected-output", type=str, required=True, help="Path to the expected output numpy file.")
    args = parser.parse_args()

    expected_output = np.load(args.expected_output)
    if expected_output.shape[0] != INPUT_DIM:
        raise ValueError(f"Expected output shape {INPUT_DIM}, got {expected_output.shape[0]}")

    interpreter = load_tflite_model(args.tflite_model)
    input_details, output_details = get_input_output_details(interpreter)

    # Assuming representative input data is available for validation
    representative_input = np.random.rand(*input_details[0]['shape']).astype(np.float32)
    
    validate_model(interpreter, representative_input, expected_output)
    print("Validation successful: TFLite model output matches expected output.")

if __name__ == "__main__":
    main()