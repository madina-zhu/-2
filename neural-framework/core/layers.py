import numpy as np
from .tensor import Tensor

class Module:
    """Базовый класс для всех слоев."""
    def parameters(self):
        return []

class Linear(Module):
    """Полносвязный слой."""
    
    def __init__(self, in_features, out_features):
        # Улучшенная инициализация Xavier
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros((1, out_features)),
            requires_grad=True
        )
    
    def parameters(self):
        return [self.weight, self.bias]
    
    def forward(self, x):
        return x @ self.weight + self.bias
    
    def __call__(self, x):
        return self.forward(x)


class Sequential(Module):
    """Контейнер для последовательного применения слоев."""
    
    def __init__(self, *layers):
        self.layers = layers
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x):
        return self.forward(x)