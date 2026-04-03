import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.tensor import Tensor
from core.layers import Linear, Sequential
from core.activations import Sigmoid
from core.losses import MSELoss
from core.optimizers import SGD

# Данные
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = Tensor(X_data)
y = Tensor(y_data)

# Модель
model = Sequential(
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 1),
    Sigmoid()
)

loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.5)

print("Проверка градиентов перед обучением:")
for i, p in enumerate(model.parameters()):
    print(f"  Parameter {i}: shape={p.data.shape}, mean={p.data.mean():.6f}, std={p.data.std():.6f}")

# Один шаг обучения
optimizer.zero_grad()
y_pred = model(X)
loss = loss_fn(y_pred, y)
loss.backward()

print("\nГрадиенты после backward:")
for i, p in enumerate(model.parameters()):
    if p.grad is not None:
        print(f"  Parameter {i}: grad_mean={p.grad.mean():.6f}, grad_std={p.grad.std():.6f}, grad_norm={np.linalg.norm(p.grad):.6f}")
    else:
        print(f"  Parameter {i}: grad is None!")

optimizer.step()

print("\nВеса после обновления:")
for i, p in enumerate(model.parameters()):
    print(f"  Parameter {i}: mean={p.data.mean():.6f}, std={p.data.std():.6f}")

# Проверка предсказания
print("\nПредсказание до обучения:")
y_pred = model(X)
print(y_pred.data.flatten())

# Небольшое обучение
print("\n--- Обучение 500 эпох ---")
for epoch in range(500):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: loss={loss.data:.6f}")

print("\nФинальные предсказания:")
y_pred = model(X)
for i in range(4):
    print(f"  {X_data[i]} -> {y_pred.data[i][0]:.4f}(ожидалось: {y_data[i][0]})")
