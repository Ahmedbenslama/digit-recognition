import numpy as np


class DenseLayer:
    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        # He initialization for weights, zero initialization for bias
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        
        # Cache variables for backpropagation
        self.inputs = None
        self.z = None

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "softmax":
            exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        elif self.activation == "none":
            return z
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Input features ({inputs.shape[1]}) do not match layer input size ({self.weights.shape[0]})."
            )

        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        return self._activate(self.z)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        if self.activation == "relu":
            dZ = output_gradient * (self.z > 0)
        elif self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(self.z, -500, 500)))
            dZ = output_gradient * s * (1.0 - s)
        elif self.activation == "tanh":
            dZ = output_gradient * (1.0 - np.tanh(self.z) ** 2)
        elif self.activation in ("softmax", "none"):
            dZ = output_gradient
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        m = self.inputs.shape[0]
        dW = np.dot(self.inputs.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.weights.T)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return dX


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient: np.ndarray, learning_rate: float):
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train_step(self, X: np.ndarray, Y_one_hot: np.ndarray, learning_rate: float) -> float:
        predictions = self.forward(X)
        loss = categorical_cross_entropy(Y_one_hot, predictions)
        loss_grad = predictions - Y_one_hot
        self.backward(loss_grad, learning_rate)
        return loss