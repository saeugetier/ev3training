from pathlib import Path
import tensorflow as tf
import numpy as np
import unittest

class TestConversion(unittest.TestCase):
    def setUp(self):
        self.checkpoint_path = Path("path/to/model_2500.pt")
        self.tflite_model_path = Path("path/to/model.tflite")
        self.input_data = np.zeros((1, 36), dtype=np.float32)  # Example input shape
        self.expected_output = np.zeros((1, 2), dtype=np.float32)  # Adjust based on model output shape

    def test_conversion(self):
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_path))
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], self.input_data)

        # Run the model
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Compare the output with expected output
        np.testing.assert_allclose(output_data, self.expected_output, rtol=1e-05, atol=1e-05)

if __name__ == "__main__":
    unittest.main()