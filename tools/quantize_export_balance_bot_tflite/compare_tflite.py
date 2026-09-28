"""Compare PyTorch, TensorFlow Lite INT8 and Rust policy outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

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


def load_pytorch_policy(checkpoint: Path) -> BalanceBotPolicyRef:
    model = BalanceBotPolicyRef()
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = _extract_state_dict(state)

    load_from_rsl_rl_state_dict(
        model,
        state_dict,
        prefix=_resolve_actor_prefix(state_dict),
    )

    normalizer = _find_normalizer(state, state_dict)
    if normalizer is not None:
        fold_input_normalization(model, *normalizer)

    model.eval()
    return model


def quantize_input(values: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]

    if dtype in (np.float32, np.float64):
        return values.astype(dtype)

    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError("TFLite input has no valid quantization scale")

    info = np.iinfo(dtype)
    quantized = np.round(values / scale + zero_point)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def dequantize_output(values: np.ndarray, detail: dict) -> np.ndarray:
    if values.dtype in (np.float32, np.float64):
        return values.astype(np.float32)

    scale, zero_point = detail["quantization"]
    if scale <= 0:
        raise ValueError("TFLite output has no valid quantization scale")

    return (values.astype(np.float32) - zero_point) * scale


def run_tflite(model_path: Path, observation: np.ndarray) -> np.ndarray:
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    if len(inputs) != 1:
        raise ValueError(f"expected one TFLite input, got {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"expected one TFLite output, got {len(outputs)}")

    input_detail = inputs[0]
    input_shape = tuple(input_detail["shape"])

    if int(np.prod(input_shape)) != INPUT_DIM:
        raise ValueError(
            f"expected TFLite input with {INPUT_DIM} values, "
            f"got shape {input_shape}"
        )

    tensor = quantize_input(observation, input_detail).reshape(input_shape)
    interpreter.set_tensor(input_detail["index"], tensor)
    interpreter.invoke()

    output_detail = outputs[0]
    output = interpreter.get_tensor(output_detail["index"])
    return dequantize_output(output, output_detail).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--obs-scale", type=float, required=True)
    parser.add_argument("--input", type=int, nargs=INPUT_DIM, required=True)
    args = parser.parse_args()

    obs_q15 = np.asarray(args.input, dtype=np.float32)
    observation = obs_q15 * args.obs_scale

    pytorch = load_pytorch_policy(args.checkpoint)
    with torch.no_grad():
        pytorch_action = (
            pytorch.act(pytorch.fc1(torch.from_numpy(observation).unsqueeze(0)))
        )
        pytorch_action = pytorch.fc_out(
            pytorch.act(
                pytorch.fc3(
                    pytorch.act(
                        pytorch.fc2(
                            pytorch.act(
                                pytorch.fc1(
                                    torch.from_numpy(observation).unsqueeze(0)
                                )
                            )
                        )
                    )
                )
            )
        )[0].numpy()

    tflite_action = run_tflite(args.tflite, observation)

    if len(tflite_action) != 2:
        raise ValueError(f"expected two actions, got {len(tflite_action)}")

    print(f"input_real={observation.tolist()}")
    print(
        "pytorch_action_raw: "
        f"left={pytorch_action[0]:.8f} right={pytorch_action[1]:.8f}"
    )
    print(
        "tflite_action_raw: "
        f"left={tflite_action[0]:.8f} right={tflite_action[1]:.8f}"
    )
    print(
        "difference: "
        f"left={tflite_action[0] - pytorch_action[0]:+.8f} "
        f"right={tflite_action[1] - pytorch_action[1]:+.8f}"
    )
    print(
        "tflite_action_clamped: "
        f"left={np.clip(tflite_action[0], -1, 1):.8f} "
        f"right={np.clip(tflite_action[1], -1, 1):.8f}"
    )


if __name__ == "__main__":
    main()