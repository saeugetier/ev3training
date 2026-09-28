from __future__ import annotations

import argparse
import os
import tensorflow as tf

def load_checkpoint(checkpoint_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(checkpoint_path)
    return model

def quantize_model(model: tf.keras.Model) -> tf.lite.TFLiteModel:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

def save_tflite_model(tflite_model: tf.lite.TFLiteModel, output_path: str) -> None:
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the RL training checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the converted TFLite model.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = load_checkpoint(args.checkpoint)
    tflite_model = quantize_model(model)
    save_tflite_model(tflite_model, args.output)
    print(f"TFLite model saved to {args.output}")

if __name__ == "__main__":
    main()