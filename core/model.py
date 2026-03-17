import numpy as np
from core.loss import Loss
from typing import List
class Model:
    """Neural Network Model orchestrator."""
    def __init__(self, layers: List, loss_fn: Loss):
        self.layers = layers
        self.loss_fn = loss_fn
        self.history = []
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Pass input through all layers."""
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backpropagate the gradient through all layers in reverse."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
