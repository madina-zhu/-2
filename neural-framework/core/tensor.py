import numpy as np

class Tensor:
    """Основной класс для хранения данных и градиентов."""
    
    def __init__(self, data, requires_grad=False, depends_on=None, creation_op=None):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.depends_on = depends_on if depends_on is not None else []
        self.creation_op = creation_op
    
    def zero_grad(self):
        """Обнуляет градиент."""
        self.grad = None
    
    def __add__(self, other):
        """Сложение."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(new_data, requires_grad, [self, other], 'add')
    
    def __mul__(self, other):
        """Поэлементное умножение."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(new_data, requires_grad, [self, other], 'mul')
    
    def __matmul__(self, other):
        """Матричное умножение."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        new_data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        return Tensor(new_data, requires_grad, [self, other], 'matmul')
    
    def __sub__(self, other):
        """Вычитание."""
        return self + (-other)
    
    def __neg__(self):
        """Отрицание."""
        return self * (-1)
    
    def __getitem__(self, idx):
        """Поддержка индексации и срезов."""
        new_data = self.data[idx]
        return Tensor(new_data, requires_grad=self.requires_grad)
    
    def mean(self):
        """Среднее значение."""
        data = np.mean(self.data)
        return Tensor(data, requires_grad=self.requires_grad, depends_on=[self], creation_op='mean')
    
    def backward(self, grad=None):
        """Обратное распространение."""
        if grad is None:
            grad = np.ones_like(self.data)
        if self.requires_grad and self.creation_op is None:
            print(f"Gradient for parameter tensor: norm = {np.linalg.norm(grad):.6f}")
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad
        
        if not self.requires_grad or not self.depends_on:
            return
        
        if self.creation_op == 'add':
            a, b = self.depends_on
            a.backward(grad)
            b.backward(grad)
        
        elif self.creation_op == 'mul':
            a, b = self.depends_on
            a.backward(grad * b.data)
            b.backward(grad * a.data)
        
        elif self.creation_op == 'matmul':
            a, b = self.depends_on
            a.backward(grad @ b.data.T)
            b.backward(a.data.T @ grad)
        
        elif self.creation_op == 'relu':
            a = self.depends_on[0]
            grad_relu = grad * (a.data > 0)
            a.backward(grad_relu)
        
        elif self.creation_op == 'sigmoid':
            a = self.depends_on[0]
            sig = self.data
            grad_sigmoid = grad * (sig * (1 - sig))
            a.backward(grad_sigmoid)
        
        elif self.creation_op == 'mean':
            a = self.depends_on[0]
            n = a.data.size
            grad_mean = grad / n * np.ones_like(a.data)
            a.backward(grad_mean)
    
    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    