# Neural Framework — Custom Neural Network Library

Лёгкий фреймворк для обучения полносвязных нейронных сетей, реализованный с нуля на NumPy.

---

## 👥 Участники

| Имя        | Вклад                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------- |
| **Катя**   | Ядро фреймворка: autograd, слои, функции потерь, оптимизаторы                                  |
| **Мадина** | Работа с данными (Dataset, DataLoader), примеры (MNIST, Iris), документация, no-code интерфейс |

---

## 🚀 Возможности

### 🧠 Ядро (Катя)

* Реализация Tensor с автоматическим дифференцированием (autograd)
* Полносвязные слои (`Linear`, `Sequential`)
* Функции активации: ReLU, Sigmoid, Softmax
* Функции потерь: MSE, CrossEntropy
* Оптимизаторы:

  * SGD
  * Momentum SGD
  * Adam
* Gradient Clipping

---

### 📊 Работа с данными (Мадина)

* Абстракция `Dataset`
* `DataLoader` с:

  * batching
  * shuffle
  * map
* Разделение данных (`split`)
* Поддержка:

  * MNIST
  * Iris
  * California Housing
* Трансформации:

  * Normalize
  * OneHot
  * Flatten

---

### 🧪 Примеры

* Классификация MNIST
* Классификация Iris
* Регрессия (Iris / Housing)

---

### 🌐 No-code интерфейс (особенность проекта)

Реализован веб-интерфейс на Streamlit, позволяющий:

* выбирать датасет
* настраивать архитектуру сети
* выбирать оптимизатор
* запускать обучение
* видеть графики обучения

👉 Без написания кода

---

## ⚡ Быстрый старт

### Установка

```bash
git clone https://github.com/madina-zhu/neural-framework.git
cd neural-framework
pip install -r requirements.txt
```

---

### Пример использования

```python
import numpy as np
from core.layers import Sequential, Linear
from core.activations import ReLU, Softmax
from core.losses import CrossEntropyLoss
from core.optimizers import Adam
from data.dataset import DataLoader, ArrayDataset

# Данные
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

dataset = ArrayDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Модель
model = Sequential([
    Linear(20, 64),
    ReLU(),
    Linear(64, 2),
    Softmax()
])

# Обучение
loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

model.fit(loader, loss, optimizer, epochs=10)
```

---

## 🧪 Примеры запуска

### MNIST

```bash
python examples/mnist_demo.py
```

### Iris

```bash
python examples/iris_demo.py
```

---

## 🌐 Web-интерфейс

```bash
cd web_demo
streamlit run app.py
```

Открыть в браузере:
http://localhost:8501

---

## 🎥 Видео

Будет добавлено позже

---

## 📁 Структура проекта

```
neural-framework/
│
├── core/        # ядро (Катя)
├── data/        # данные (Мадина)
├── examples/    # примеры (Мадина)
├── web_demo/    # Streamlit (Мадина)
├── tests/
└── README.md
```

---

## ⭐ Особенности

* Реализация autograd с нуля
* Собственный DataLoader (как в PyTorch)
* No-code интерфейс
* Простое обучение "в несколько строк"

---

## 🏁 Итог

Фреймворк позволяет:

* быстро создавать нейросети
* обучать их на реальных данных
* визуализировать обучение
* использовать без программирования (через UI)

---
