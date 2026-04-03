import numpy as np
from .tensor import Tensor

class MSELoss:
    """Mean Squared Error Loss для регрессии."""
    
    def __call__(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = (diff * diff).mean()
        return loss


class CrossEntropyLoss:
    """Cross Entropy Loss для классификации."""
    
    def __call__(self, y_pred, y_true):
        shifted = y_pred.data - np.max(y_pred.data, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        
        batch_size = y_pred.data.shape[0]
        y_true_flat = y_true.data.astype(int).flatten()
        correct_probs = probs[np.arange(batch_size), y_true_flat]
        loss_val = -np.log(correct_probs + 1e-7).mean()
        
        loss = Tensor(loss_val, requires_grad=True)
        
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), y_true_flat] = 1
        grad_logits = (probs - one_hot) / batch_size
        
        def custom_backward(grad):
            y_pred.backward(grad * grad_logits)
        
        loss.backward = custom_backward.__get__(loss)
        return loss
