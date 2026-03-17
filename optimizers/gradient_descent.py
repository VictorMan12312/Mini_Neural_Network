from optimizers.base_optimizer import BaseOptimizer
class GradientDescent(BaseOptimizer):
    """Standard Gradient Descent Optimizer."""
    def __init__(self, lr: float = 0.01):
        self.lr = lr
    def step(self, layers):
        for layer in layers:
            # Update only layers with learnable parameters (Dense)
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b