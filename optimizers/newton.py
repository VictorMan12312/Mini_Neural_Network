import numpy as np
from optimizers.base_optimizer import BaseOptimizer
class NewtonOptimizer(BaseOptimizer):
    """
    Newton-Raphson optimizer attempting second-order convergence.
    
    LIMITATIONS:
    Computing the exact Hessian matrix scales O(N^2) where N is the number of parameters. 
    Here, a simplified diagonal approximation of the Hessian combined with a pseudo-inverse 
    is implemented to prevent memory exhaustion and singular matrices on deep networks.
    """
    def __init__(self, lr: float = 1.0, epsilon: float = 1e-4):
        self.lr = lr
        self.epsilon = epsilon
    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                # Reshape gradients into 1D vectors for Hessian compatibility
                grad_W_flat = layer.grad_W.flatten()
                
                # Approximate the Hessian using a localized identity-regularized proxy.
                # A true formulation would track secant equations (like BFGS) or exact 2nd derivatives.
                H_approx_W = np.eye(len(grad_W_flat)) * (np.linalg.norm(grad_W_flat) + self.epsilon)
                
                # Compute Pseudo-Inverse (pinv handles instability)
                H_inv_W = np.linalg.pinv(H_approx_W)
                
                # Newton Step: W_new = W_old - lr * (H^-1 * grad)
                step_W = H_inv_W.dot(grad_W_flat).reshape(layer.W.shape)
                layer.W -= self.lr * step_W
                
                # Bias scaling
                grad_b_flat = layer.grad_b.flatten()
                H_approx_b = np.eye(len(grad_b_flat)) * (np.linalg.norm(grad_b_flat) + self.epsilon)
                H_inv_b = np.linalg.pinv(H_approx_b)
                
                step_b = H_inv_b.dot(grad_b_flat).reshape(layer.b.shape)
                layer.b -= self.lr * step_b