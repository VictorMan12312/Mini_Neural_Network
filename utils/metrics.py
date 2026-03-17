import numpy as np
from typing import Tuple
def accuracy(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> float:
    preds = (y_pred >= threshold).astype(int)
    return float(np.mean(preds == y_true))
def precision_recall(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
    preds = (y_pred >= threshold).astype(int)
    
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return float(precision), float(recall)