import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tensor import Tensor
from core.layers import Linear, Sequential
from core.activations import Sigmoid
from core.losses import MSELoss
from core.optimizers import SGD

# Улучшенный даталоадер, который работает с Tensor
class SimpleDataLoader:
    def __init__(self, X, y, batch_size=2, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.data.shape[0]
        
        # Создаем индексы
        self.indices = list(range(self.n_samples))
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            
            # Собираем батч через индексы
            X_batch_data = np.array([self.X.data[idx] for idx in batch_indices])
            y_batch_data = np.array([self.y.data[idx] for idx in batch_indices])
            
            X_batch = Tensor(X_batch_data)
            y_batch = Tensor(y_batch_data)
            
            yield X_batch, y_batch
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

# Данные для XOR задачи
X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Оборачиваем в Tensor
X = Tensor(X_data)
y = Tensor(y_data)

# Создаем модель
model = Sequential(
    Linear(2, 4),
    Sigmoid(),
    Linear(4, 1),
    Sigmoid()
)

loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.5)
train_loader = SimpleDataLoader(X, y, batch_size=2)

print("Начинаем обучение...")
print("=" * 40)

# Обучаем вручную с выводом каждые 100 эпох
for epoch in range(1000):
    total_loss = 0.0
    batches = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        batches += 1
    
    avg_loss = total_loss / batches
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}/1000, Loss: {avg_loss:.6f}")

print("=" * 40)
print("\nРезультаты после обучения:")

# Проверка результатов
for i in range(4):
    pred = model(Tensor(X_data[i:i+1]))
    print(f"Вход: {X_data[i]} -> Предсказание: {pred.data[0][0]:.4f} (ожидалось: {y_data[i][0]})")

# Проверка точности
predictions = []
for i in range(4):
    pred = model(Tensor(X_data[i:i+1]))
    predictions.append(round(pred.data[0][0]))
    
accuracy = sum(1 for i in range(4) if predictions[i] == y_data[i][0]) / 4 * 100
print(f"\nТочность: {accuracy:.0f}%")
