import numpy as np
class Layer:
    """Base layer abstract class."""
    def forward(self, inputs):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError
class Dense(Layer):
    """Fully connected dense layer."""
    def __init__(self, input_size: int, output_size: int):
        # Initialize weights using He initialization (scaled by sqrt(2/input_size))
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.inputs = None
        self.grad_W = None
        self.grad_b = None
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Gradients w.r.t parameters
        self.grad_W = np.dot(self.inputs.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        # Gradient w.r.t input to pass down to previous layer
        return np.dot(grad_output, self.W.T)
class ReLU(Layer):
    """Rectified Linear Unit activation."""
    def __init__(self):
        self.inputs = None
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output.copy()
        grad_input[self.inputs <= 0] = 0
        return grad_input
class Sigmoid(Layer):
    """Sigmoid activation function."""
    def __init__(self):
        self.outputs = None
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Clip inputs to prevent overflow warnings in exp
        inputs_clipped = np.clip(inputs, -500, 500)
        self.outputs = 1.0 / (1.0 + np.exp(-inputs_clipped))
        return self.outputs
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        return grad_output * self.outputs * (1.0 - self.outputs)