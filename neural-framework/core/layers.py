import numpy as np
from .tensor import Tensor


class Module:
    """Базовый класс для всех слоев."""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


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
        self.input = None

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x):
        self.input = x
        # Если x - numpy массив, используем его напрямую
        if isinstance(x, np.ndarray):
            return x @ self.weight.data + self.bias.data
        # Если x - Tensor
        return x.data @ self.weight.data + self.bias.data

    def backward(self, grad):
        # grad - градиент от следующего слоя (batch_size, out_features)
        # Вычисляем градиенты для весов и bias
        if isinstance(self.input, np.ndarray):
            input_data = self.input
        else:
            input_data = self.input.data

        # grad_weight: (in_features, out_features) = input.T @ grad
        grad_weight = input_data.T @ grad
        # grad_bias: (1, out_features) = sum(grad по батчу)
        grad_bias = grad.sum(axis=0, keepdims=True)
        # grad_input: (batch_size, in_features) = grad @ weight.T
        grad_input = grad @ self.weight.data.T

        self.weight.grad = grad_weight
        self.bias.grad = grad_bias

        return grad_input

    def __call__(self, x):
        return self.forward(x)


class Sequential(Module):
    """Контейнер для последовательного применения слоев."""

    def __init__(self, layers):
        # Принимает список слоёв или отдельные слои
        if isinstance(layers, (list, tuple)):
            self.layers = layers
        else:
            self.layers = [layers]

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        """Обнуляет градиенты всех параметров."""
        for param in self.parameters():
            param.zero_grad()

    def fit(self, train_loader, loss_fn, optimizer, epochs=10, verbose=True, val_loader=None):
        """Метод для обучения модели."""
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Обучение
            total_loss = 0
            for X_batch, y_batch in train_loader:
                # Forward
                pred = self.forward(X_batch)
                loss = loss_fn.forward(pred, y_batch)

                # Backward
                self.zero_grad()
                grad = loss_fn.backward()
                self.backward(grad)

                # Update
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)

            # Валидация
            if val_loader:
                val_loss = 0
                for X_batch, y_batch in val_loader:
                    pred = self.forward(X_batch)
                    loss = loss_fn.forward(pred, y_batch)
                    val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                history['val_loss'].append(avg_val_loss)

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_loss:.4f} - val_loss: {avg_val_loss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_loss:.4f}")

        return history

    def predict(self, X):
        """Предсказание для новых данных."""
        return self.forward(X)

    def evaluate(self, loader, loss_fn):
        """Оценка модели на данных."""
        val_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            pred = self.forward(X_batch)
            loss = loss_fn.forward(pred, y_batch)
            val_loss += loss.item()

            # Для классификации
            if pred.shape[1] > 1:  # многоклассовая
                pred_classes = np.argmax(pred, axis=1)
                if y_batch.ndim > 1:
                    true_classes = np.argmax(y_batch, axis=1)
                else:
                    true_classes = y_batch
                correct += np.sum(pred_classes == true_classes)
                total += len(y_batch)

        avg_loss = val_loss / len(loader)
        accuracy = correct / total if total > 0 else 0

        return avg_loss, accuracy

    def __call__(self, x):
        return self.forward(x)