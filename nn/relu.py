import numpy as np
from typing import Optional

class ReLU:
    def __init__(self) -> None:
        self.inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        dZ = output_gradient * (self.inputs > 0)
        return dZ
