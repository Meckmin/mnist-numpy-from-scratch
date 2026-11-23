import numpy as np
from .softmax_cross_entropy import SoftmaxCrossEntropy

class Model:
    def __init__(self, layers: list, loss: SoftmaxCrossEntropy) -> None:
        self.layers = layers
        self.loss = loss

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self.loss.forward(self.forward(X), Y)

    def backward(self) -> np.ndarray:
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)
        return np.argmax(logits, axis=1)



