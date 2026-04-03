import numpy as np
from .tensor import Tensor

class Module:
    """Базовый класс для всех слоев и активаций."""
    def parameters(self):
        return []  # По умолчанию нет параметров

class ReLU(Module):
    """ReLU активация: max(0, x)"""
    
    def forward(self, x):
        data = np.maximum(0, x.data)
        return Tensor(data, requires_grad=x.requires_grad, depends_on=[x], creation_op='relu')
    
    def __call__(self, x):
        return self.forward(x)


class Sigmoid(Module):
    """Sigmoid активация: 1 / (1 + e^{-x})"""
    
    def forward(self, x):
        data = 1 / (1 + np.exp(-x.data))
        return Tensor(data, requires_grad=x.requires_grad, depends_on=[x], creation_op='sigmoid')
    
    def __call__(self, x):
        return self.forward(x)


class Softmax(Module):
    """Softmax активация для классификации."""
    
    def forward(self, x):
        shifted = x.data - np.max(x.data, axis=1, keepdims=True)
        exp = np.exp(shifted)
        data = exp / np.sum(exp, axis=1, keepdims=True)
        return Tensor(data, requires_grad=x.requires_grad, depends_on=[x], creation_op='softmax')
    
    def __call__(self, x):
        return self.forward(x)
