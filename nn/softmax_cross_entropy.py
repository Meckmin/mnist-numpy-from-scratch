import numpy as np
from typing import Optional

class SoftmaxCrossEntropy:
    def __init__(self) -> None :
        self.y_hat : Optional[np.ndarray] = None
        self.y_true : Optional[np.ndarray] = None

    def forward(self, Z: np.ndarray, y_true: np.ndarray) -> float:
        self.y_true = y_true

        # Preventing overflow
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))

        # Calculating the result of softmax function
        y_hat = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self.y_hat = y_hat

        # Testing if row sums add up to 1
        y_hat_sum = np.sum(y_hat, axis=1, keepdims=True)
        assert np.allclose(y_hat_sum, 1.0), f"Softmax row sums do not add up to 1: {y_hat_sum.ravel()}"

        # Clipping and preventing nan values
        y_hat_clipped = np.clip(y_hat, 1e-12, 1.0)

        # Calculating loss
        L = -np.sum((self.y_true * np.log(y_hat_clipped)), axis=1)

        # Calculating and returning cost
        cost = np.mean(L)
        return cost

    def backward(self) -> np.ndarray:
        m = self.y_true.shape[0]
        dZ = (self.y_hat - self.y_true) / m
        return dZ