import numpy as np
import data_loader
from nn.dense import Dense
from nn.model import Model
from nn.relu import ReLU
from nn.softmax_cross_entropy import SoftmaxCrossEntropy
from train import train

def save(model, path) -> None:
    layers_with_param = {}

    count = 1
    for layer in model.layers:
        if hasattr(layer, "W"):
            layers_with_param[f"W{count}"] = layer.W
            layers_with_param[f"b{count}"] = layer.b
            count += 1
    np.savez(path, **{k: v.astype(np.float32) for k, v in layers_with_param.items()})

def load(path) -> Model:
    # assumes a relu between every pair of consecutive dense layers and no activation after the last one
    layers = []
    with np.load(path) as data:
        for param in data:
            weight = data[param]
            if "W" in param:
                if len(layers) != 0:
                    layers.append(ReLU())
                layers.append(Dense(*weight.shape))
        new_model = Model(layers, SoftmaxCrossEntropy())

        count = 1
        for layer in new_model.layers:
            if hasattr(layer, "W"):
                layer.W = data[f"W{count}"]
                layer.b = data[f"b{count}"]
                count += 1
    return new_model

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_loader.load_mnist()
    model, history = train(X_train, Y_train, X_val, Y_val, 0.5, 128, 20)

    save(model, "weights/model.npz")
    new_model = load("weights/model.npz")

    test_acc = 100 * np.mean(model.predict(X_test) == np.argmax(Y_test, axis=1))
    new_test_acc = 100 * np.mean(new_model.predict(X_test) == np.argmax(Y_test, axis=1))

    assert test_acc == new_test_acc, "Different test accuracies"
    assert np.allclose(new_model.forward(X_val), model.forward(X_val), atol=1e-5), "Different logits"

    diff = np.abs(new_model.forward(X_val) - model.forward(X_val)).max()
    print(f"Test logit differences: {diff}")
    print(f"Test dataset accuracy: %{test_acc}")
