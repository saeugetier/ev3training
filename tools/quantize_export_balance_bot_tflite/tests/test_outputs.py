from pathlib import Path
import numpy as np
import tensorflow as tf
import unittest

class TestTensorFlowLiteOutputs(unittest.TestCase):
    def setUp(self):
        self.tflite_model_path = Path("path/to/your/model.tflite")
        self.input_data = np.array([0] * 36, dtype=np.int16)  # Replace with actual test input
        self.expected_outputs = {
            'fc1': [0] * 8,  # Replace with expected output for fc1
            'fc2': [0] * 8,  # Replace with expected output for fc2
            'fc3': [0] * 8,  # Replace with expected output for fc3
            'action': [0.0, 0.0]  # Replace with expected action output
        }

    def load_tflite_model(self):
        interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model_path))
        interpreter.allocate_tensors()
        return interpreter

    def test_outputs(self):
        interpreter = self.load_tflite_model()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], self.input_data)
        interpreter.invoke()

        fc1_output = interpreter.get_tensor(output_details[0]['index'])
        fc2_output = interpreter.get_tensor(output_details[1]['index'])
        fc3_output = interpreter.get_tensor(output_details[2]['index'])
        action_output = interpreter.get_tensor(output_details[3]['index'])

        # Compare outputs
        np.testing.assert_array_almost_equal(fc1_output[:8], self.expected_outputs['fc1'], decimal=2)
        np.testing.assert_array_almost_equal(fc2_output[:8], self.expected_outputs['fc2'], decimal=2)
        np.testing.assert_array_almost_equal(fc3_output[:8], self.expected_outputs['fc3'], decimal=2)
        np.testing.assert_array_almost_equal(action_output, self.expected_outputs['action'], decimal=2)

if __name__ == "__main__":
    unittest.main()