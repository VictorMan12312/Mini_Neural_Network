import numpy as np
class Loss:
    """Base Loss class."""
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def backward(self, y_pred, y_true):
        raise NotImplementedError
class MSE(Loss):
    """Mean Squared Error Loss."""
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean(np.power(y_true - y_pred, 2)))
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size
class BCE(Loss):
    """Binary Cross Entropy Loss."""
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Clip to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)))
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        # Derivative of BCE
        return ((y_pred_clipped - y_true) / (y_pred_clipped * (1.0 - y_pred_clipped))) / y_true.shape[0]
