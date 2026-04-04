import numpy as np


class Tensor:
    """Основной класс для хранения данных и градиентов."""

    def __init__(self, data, requires_grad=False, depends_on=None, creation_op=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self.depends_on = depends_on if depends_on is not None else []
        self.creation_op = creation_op

    def zero_grad(self):
        """Обнуляет градиент."""
        self.grad = None

    def item(self):
        """Возвращает скалярное значение для тензора с одним элементом."""
        return float(self.data.item())

    def __add__(self, other):
        """Операция сложения."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            depends_on=[self, other],
            creation_op='add'
        )

    def __mul__(self, other):
        """Операция умножения."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            depends_on=[self, other],
            creation_op='mul'
        )

    def __matmul__(self, other):
        """Операция матричного умножения."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            depends_on=[self, other],
            creation_op='matmul'
        )

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
