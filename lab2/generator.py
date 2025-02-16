import numpy as np
import pandas as pd

# Параметры нормального распределения
age_mean, age_std = 45, 3
salary_mean, salary_std = 50000, 5000

# Генерация данных
np.random.seed(42)  # Для воспроизводимости
ages = np.random.normal(age_mean, age_std, 100)
salaries = np.random.normal(salary_mean, salary_std, 100)

# Ограничение значений
ages = np.clip(ages, 30, 50).astype(int)
salaries = np.clip(salaries, 25000, 75000).astype(int)

# Генерация целевой переменной (Bought Iphone 14)
# Предположим, что вероятность покупки зависит от возраста и зарплаты
probabilities = 1 / (1 + np.exp(-0.01 * (ages - 40) + 0.0001 * (salaries - 50000)))
bought_iphone = np.random.binomial(1, probabilities)

# Создание DataFrame
data = pd.DataFrame({
    'Age': ages,
    'Salary': salaries,
    'Bought Iphone 14': bought_iphone
})

# Сохранение в CSV
data.to_csv('gnb2.csv', index=False)