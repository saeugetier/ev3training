from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from tools.quantize_export_balance_bot.cli import (
    _extract_state_dict,
    _find_normalizer,
    _resolve_actor_prefix,
)
from tools.quantize_export_balance_bot.network_spec import INPUT_DIM
from tools.quantize_export_balance_bot.reference_model import (
    BalanceBotPolicyRef,
    fold_input_normalization,
    load_from_rsl_rl_state_dict,
)


def load_checkpoint(checkpoint: Path) -> BalanceBotPolicyRef:
    model = BalanceBotPolicyRef()
    state = torch.load(checkpoint, map_location="cpu")

    state_dict = _extract_state_dict(state)
    prefix = _resolve_actor_prefix(state_dict)
    load_from_rsl_rl_state_dict(model, state_dict, prefix=prefix)

    normalizer = _find_normalizer(state, state_dict)
    if normalizer is not None:
        fold_input_normalization(model, *normalizer)
    else:
        print("warning: observation normalizer was not found")

    model.eval()
    return model


def detect_activation(model: BalanceBotPolicyRef) -> tuple[str, float]:
    """Detect the activation used by BalanceBotPolicyRef.act."""
    values = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]])

    with torch.no_grad():
        result = model.act(values).detach().cpu().numpy()[0]

    if np.allclose(result, np.where(values.numpy()[0] < 0, values.numpy()[0] * 0.01, values.numpy()[0])):
        return "leaky_relu", 0.01

    if np.allclose(result, np.tanh(values.numpy()[0]), atol=1e-6):
        return "tanh", 0.0

    if np.allclose(result, np.maximum(values.numpy()[0], 0.0), atol=1e-6):
        return "relu", 0.0

    if np.allclose(result, values.numpy()[0], atol=1e-6):
        return "linear", 0.0

    raise ValueError(
        "Unsupported activation in BalanceBotPolicyRef.act. "
        f"Observed values: {result}"
    )


def create_tf_model(model: BalanceBotPolicyRef) -> tf.keras.Model:
    activation, alpha = detect_activation(model)

    inputs = tf.keras.Input(
        shape=(INPUT_DIM,),
        batch_size=1,
        dtype=tf.float32,
        name="observation",
    )

    x = inputs

    for layer_name in ("fc1", "fc2", "fc3"):
        source_layer = getattr(model, layer_name)

        x = tf.keras.layers.Dense(
            int(source_layer.out_features),
            activation=None,
            name=layer_name,
        )(x)

        # compare_policy.py aktiviert fc1, fc2 und fc3.
        if activation == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(
                negative_slope=alpha,
                name=f"{layer_name}_activation",
            )(x)
        elif activation == "relu":
            x = tf.keras.layers.ReLU(
                name=f"{layer_name}_activation",
            )(x)
        elif activation == "tanh":
            x = tf.keras.layers.Activation(
                "tanh",
                name=f"{layer_name}_activation",
            )(x)
        elif activation != "linear":
            raise ValueError(f"Unsupported TensorFlow activation: {activation}")

    output = tf.keras.layers.Dense(
        int(model.fc_out.out_features),
        activation=None,
        name="action",
    )(x)

    tf_model = tf.keras.Model(
        inputs=inputs,
        outputs=output,
        name="balance_bot_policy",
    )

    # Modell bauen, bevor die Gewichte gesetzt werden.
    tf_model(tf.zeros((1, INPUT_DIM), dtype=tf.float32))

    for layer_name in ("fc1", "fc2", "fc3"):
        torch_layer = getattr(model, layer_name)
        tf_layer = tf_model.get_layer(layer_name)

        tf_layer.set_weights(
            [
                torch_layer.weight.detach().cpu().numpy().T.astype(np.float32),
                torch_layer.bias.detach().cpu().numpy().astype(np.float32),
            ]
        )

    output_layer = tf_model.get_layer("action")
    output_layer.set_weights(
        [
            model.fc_out.weight.detach().cpu().numpy().T.astype(np.float32),
            model.fc_out.bias.detach().cpu().numpy().astype(np.float32),
        ]
    )

    tf_model.trainable = False
    return tf_model


def representative_dataset(
    samples: int = 1024,
    seed: int = 1234,
):
    """Representative observations in the same real-value domain as PyTorch."""
    rng = np.random.default_rng(seed)

    # Zusätzlich Nullwerte kalibrieren, da sie auch vom Rust policy_probe
    # häufig verwendet werden.
    yield [np.zeros((1, INPUT_DIM), dtype=np.float32)]

    for _ in range(max(0, samples - 1)):
        observation = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(1, INPUT_DIM),
        ).astype(np.float32)
        yield [observation]


def convert_to_tflite(
    checkpoint: Path,
    tflite_model_path: Path,
    calibration_samples: int = 1024,
) -> None:
    torch_model = load_checkpoint(checkpoint)
    tf_model = create_tf_model(torch_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        samples=calibration_samples,
    )

    # Erzwingt vollständig quantisierte INT8-Ein-/Ausgaben.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_model_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_model_path.write_bytes(tflite_model)

    print(f"written: {tflite_model_path}")
    print(f"size: {len(tflite_model)} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an RL PyTorch checkpoint as a fully INT8 TFLite model.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1024,
        help="Number of representative calibration observations.",
    )
    args = parser.parse_args()

    convert_to_tflite(
        checkpoint=args.checkpoint,
        tflite_model_path=args.output,
        calibration_samples=args.calibration_samples,
    )


if __name__ == "__main__":
    main()