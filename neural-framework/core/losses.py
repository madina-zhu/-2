import numpy as np
from .tensor import Tensor


class MSELoss:
    """Mean Squared Error loss для регрессии."""

    def __init__(self):
        self.pred = None
        self.target = None

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        # Если pred - Tensor, используем его данные
        if hasattr(pred, 'data'):
            pred_data = np.array(pred.data, dtype=np.float32)
        else:
            pred_data = np.array(pred, dtype=np.float32)

        # Если target - Tensor, используем его данные
        if hasattr(target, 'data'):
            target_data = np.array(target.data, dtype=np.float32)
        else:
            target_data = np.array(target, dtype=np.float32)

        # Убеждаемся, что формы совпадают
        if pred_data.shape != target_data.shape:
            target_data = target_data.reshape(pred_data.shape)

        loss_data = np.mean((pred_data - target_data) ** 2)
        loss = Tensor(loss_data, requires_grad=True)

        # Сохраняем для backward
        self.pred = pred_data
        self.target = target_data
        return loss

    def backward(self):
        # Градиент MSE loss
        grad = 2 * (self.pred - self.target) / self.pred.size
        return grad.astype(np.float32)


class CrossEntropyLoss:
    """Cross Entropy loss для классификации."""

    def __init__(self):
        self.pred = None
        self.target = None

    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        # Если pred - Tensor, используем его данные
        if hasattr(pred, 'data'):
            pred_data = np.array(pred.data, dtype=np.float32)
        else:
            pred_data = np.array(pred, dtype=np.float32)

        # Преобразуем target в one-hot если нужно
        if hasattr(target, 'data'):
            target_data = np.array(target.data, dtype=np.float32)
        else:
            target_data = np.array(target, dtype=np.float32)

        if target_data.ndim == 1 or target_data.shape[1] == 1:
            # target - индексы классов
            n_samples = len(target_data)
            n_classes = pred_data.shape[1]
            target_one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
            target_one_hot[np.arange(n_samples), target_data.astype(int)] = 1
            target_data = target_one_hot

        # Численно стабильный расчет
        pred_data = np.clip(pred_data, 1e-15, 1 - 1e-15)
        loss_data = -np.mean(np.sum(target_data * np.log(pred_data), axis=1))

        loss = Tensor(loss_data, requires_grad=True)

        # Сохраняем для backward
        self.pred = pred_data
        self.target = target_data
        return loss

    def backward(self):
        # Градиент Cross Entropy loss
        grad = self.pred - self.target
        return grad.astype(np.float32)