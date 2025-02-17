import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загружаем датасет
dataset = pd.read_csv('dt.csv')
# Удаление записей с пропущенными значениями
dataset.dropna(inplace=True)
# Удаление дубликатов
dataset.drop_duplicates(inplace=True)

category_mapping = {
    "Instagram": 0,
    "WhatsApp": 1,
    "Safari": 2,
    "Netflix": 3,
    "Facebook": 4,
    "LinkedIn": 5
}
# Присваивает категориям числовые значения
dataset['App'] = dataset['App'].map(category_mapping)

X = dataset[['Usage (minutes)', 'Notifications', 'Times Opened']]
y = dataset['App']

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train")
print(X_train)
print("===========================================================")

print("y_train")
print(y_train)
print("===========================================================")

print("X_test")
print(X_test)
print("===========================================================")

print("y_test")
print(y_test)
print("===========================================================")

# Обучаем модель Гауссового Наивного Байеса
# Обучаем модель дерева решений
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Делаем предсказания
y_pred = classifier.predict(X_test)

# Предсказание для человека 62 лет с зарплатой 20000 на покупку телефона
print("===========================================================")
print("В каком приложении сидит человек с такими параметрами ")
print(classifier.predict(pd.DataFrame([[7, 6, 4]], columns=['Usage (minutes)', 'Notifications', 'Times Opened'])))
print("===========================================================")


# Выводим точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Вычисляем confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Вычисляем precision, recall, f1-score
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Визуализация дерева решений
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(classifier, feature_names=['Usage (minutes)', 'Notifications', 'Times Opened'], class_names=list(category_mapping.keys()), filled=True)
plt.show()