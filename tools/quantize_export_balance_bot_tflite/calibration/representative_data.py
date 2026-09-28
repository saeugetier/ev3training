from typing import List
import numpy as np

def representative_data_gen() -> List[np.ndarray]:
    # Generate representative data for calibration
    # This function should yield batches of representative data
    # For example, you can load data from a dataset or generate synthetic data
    for _ in range(100):  # Adjust the range for the number of samples
        # Replace this with actual data generation logic
        yield np.random.randint(-32768, 32767, size=(36,), dtype=np.int16).reshape(1, -1)