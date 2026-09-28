from __future__ import annotations

import argparse
import tensorflow as tf
import torch
from pathlib import Path

from tools.quantize_export_balance_bot.cli import _extract_state_dict, _resolve_actor_prefix
from tools.quantize_export_balance_bot.reference_model import BalanceBotPolicyRef, load_from_rsl_rl_state_dict

def load_checkpoint(checkpoint: Path) -> BalanceBotPolicyRef:
    model = BalanceBotPolicyRef()
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(state)
    prefix = _resolve_actor_prefix(state_dict)
    load_from_rsl_rl_state_dict(model, state_dict, prefix=prefix)
    model.eval()
    return model

def convert_to_tflite(checkpoint: Path, tflite_model_path: Path) -> None:
    model = load_checkpoint(checkpoint)

    # Convert the PyTorch model to TensorFlow
    tf_model = tf.keras.models.Sequential()
    # Add layers to tf_model based on the architecture of BalanceBotPolicyRef
    # This is a placeholder; actual implementation will depend on the model architecture
    # tf_model.add(tf.keras.layers.Dense(...)) 

    # Set the model to inference mode
    tf_model.trainable = False

    # Convert the model to INT8
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    convert_to_tflite(args.checkpoint, args.output)

if __name__ == "__main__":
    main()