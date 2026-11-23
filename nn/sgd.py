class SGD:
    def __init__(self, lr: float) -> None:
        self.lr: float = lr

    def step(self, layers: list) -> None:
        for layer in layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

