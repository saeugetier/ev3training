# quantize_export_balance_bot_tflite/README.md

# Quantize Export Balance Bot

This project provides tools for converting reinforcement learning (RL) training checkpoints into TensorFlow Lite (TFLite) models in INT8 format. It includes functionalities for comparing outputs between PyTorch models and Rust implementations, ensuring that the quantized models maintain accuracy.

## Project Structure

- `__init__.py`: Marks the directory as a Python package.
- `cli.py`: Command-line interface for model conversion and validation.
- `compare_policy.py`: Compares outputs of PyTorch models with Rust implementations for Q15 observations.
- `convert_tflite.py`: Converts RL training checkpoints into INT8 TensorFlow Lite models.
- `network_spec.py`: Defines the neural network architecture specifications.
- `reference_model.py`: Contains the reference model definition for comparison.
- `validate_tflite.py`: Validates TFLite model outputs against expected results.
- `calibration/representative_data.py`: Provides representative data for model calibration during quantization.
- `tests/test_conversion.py`: Unit tests for the model conversion process.
- `tests/test_outputs.py`: Tests comparing outputs of the TFLite model with the original model.
- `requirements.txt`: Lists project dependencies.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd quantize_export_balance_bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To convert a PyTorch model checkpoint to a TensorFlow Lite model, use the following command:

```
python -m tools.quantize_export_balance_bot.convert_tflite --checkpoint path/to/model.pt
```

To compare the outputs of the PyTorch model with the Rust implementation, run:

```
python -m tools.quantize_export_balance_bot.compare_policy --checkpoint path/to/model.pt --input <input_values>
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.