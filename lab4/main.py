import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv

df = read_csv("cancer.csv")

#выводим информацию о датасете для анализа
print(df.head())
print("===============================")
print(df.info())
print("===============================")

#выводим график значений метки и количества обьектов принадлежащих конкретному значению
plt.figure(figsize=(6, 4))
sns.countplot(x="Diagnosis", hue="Diagnosis", data=df, palette={"M": "red", "B": "blue"}, legend=False)

plt.xlabel("Диагноз")
plt.ylabel("Количество случаев")
plt.title("Распределение диагнозов")

plt.show()

#матрица коррелляции
# Матрица корреляции
corr_matrix = df.corr(numeric_only=True)

# Находим признаки с высокой корреляцией (>0.9)
high_corr = corr_matrix.abs() > 0.9

# Убираем коррелирующие признаки (оставляем один из пары)
to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if high_corr.iloc[i, j]:
            colname = corr_matrix.columns[i]
            to_drop.add(colname)

# Убираем высоко коррелированные признаки
df = df.drop(columns=to_drop)
print(f"Удалены признаки: {to_drop}")
corr_matrix = df.corr(numeric_only=True)

print(corr_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляционная матрица")
plt.show()



#b - доброкачественная
#m - злокачественная
#Class (0 = нормальная транзакция, 1 = мошенничество).