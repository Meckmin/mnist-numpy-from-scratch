import numpy as np
from typing import Optional

class Dense:
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim

        self.inputs: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

        self.W: np.ndarray = np.random.randn(self.input_dim, self.output_dim) \
                             * np.sqrt(2/self.input_dim)
        self.b: np.ndarray = np.zeros((1, output_dim))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        Z = inputs @ self.W + self.b
        return Z

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        self.dW = self.inputs.T @ output_gradient
        self.db = np.sum(output_gradient, axis=0, keepdims=True)
        dX = output_gradient @ self.W.T
        return dX
