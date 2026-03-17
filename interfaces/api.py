from core.model import Model
from core.loss import MSE, BCE
from optimizers.gradient_descent import GradientDescent
from optimizers.newton import NewtonOptimizer
class APIModel(Model):
    """High-level abstraction exposing a user-friendly `.train()` syntax."""
    def __init__(self, layers, loss_type='bce'):
        loss_fn = BCE() if loss_type == 'bce' else MSE()
        super().__init__(layers, loss_fn)
        
    def train(self, X, y, optimizer_name="gd", epochs=100, lr=0.01, verbose=True):
        if optimizer_name.lower() == "gd":
            opt = GradientDescent(lr=lr)
        elif optimizer_name.lower() == "newton":
            opt = NewtonOptimizer(lr=lr)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")
            
        for epoch in range(1, epochs + 1):
            # 1. Forward pass
            y_pred = self.forward(X)
            
            # 2. Compute Loss
            loss = self.loss_fn.forward(y_pred, y)
            self.history.append(loss)
            
            # 3. Backward pass
            grad = self.loss_fn.backward(y_pred, y)
            self.backward(grad)
            
            # 4. Optimizer Step
            opt.step(self.layers)
            
            # 5. Logging
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch:04d}/{epochs} | Loss: {loss:.4f}")