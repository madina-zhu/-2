import numpy as np
from .tensor import Tensor


class ReLU:
    """ReLU активация."""

    def forward(self, x):
        # Если x - numpy массив, оборачиваем в Tensor
        if isinstance(x, np.ndarray):
            return np.maximum(0, x)
        # Если x - Tensor, используем его данные
        data = np.maximum(0, x.data)
        return Tensor(data, requires_grad=x.requires_grad, depends_on=[x], creation_op='relu')

    def __call__(self, x):
        return self.forward(x)


class Sigmoid:
    """Sigmoid активация."""

    def forward(self, x):
        if isinstance(x, np.ndarray):
            return 1 / (1 + np.exp(-x))
        data = 1 / (1 + np.exp(-x.data))
        return Tensor(data, requires_grad=x.requires_grad, depends_on=[x], creation_op='sigmoid')

    def __call__(self, x):
        return self.forward(x)


class Tanh:
    """Tanh активация."""

    def forward(self, x):
        if isinstance(x, np.ndarray):
            return np.tanh(x)
        data = np.tanh(x.data)
        return Tensor(data, requires_grad=x.requires_grad, depends_on=[x], creation_op='tanh')

    def __call__(self, x):
        return self.forward(x)


class Softmax:
    """Softmax активация для многоклассовой классификации."""

    def forward(self, x):
        if isinstance(x, np.ndarray):
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        # Для Tensor
        data = x.data
        exp_x = np.exp(data - np.max(data, axis=-1, keepdims=True))
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(softmax_output, requires_grad=x.requires_grad, depends_on=[x], creation_op='softmax')

    def __call__(self, x):
        return self.forward(x)
