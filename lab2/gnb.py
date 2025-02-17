import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загружаем датасет
dataset = pd.read_csv('gnb_oner.csv')
X = dataset[['Age', 'Salary']]
y = dataset['Bought Iphone 14']

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

# Проверка на нормальное распределение возраста
sns.histplot(X['Age'], bins=40, kde=True)
plt.xlabel("Age")
plt.show()

# Проверка на нормальное распределение зарплаты
sns.histplot(X['Salary'], bins=40, kde=True)
plt.xlabel("Salary")
plt.show()

# Обучаем модель Гауссового Наивного Байеса
# classifier = MultinomialNB()
# classifier = BernoulliNB()
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Делаем предсказания
y_pred = classifier.predict(X_test)

# Предсказание для человека 62 лет с зарплатой 20000 на покупку телефона
print("===========================================================")
print("Купит ли человек 62 лет с зарплатой 20000")
print(classifier.predict(pd.DataFrame([[32, 60000]], columns=['Age', 'Salary'])))
print("===========================================================")

# Выводим точность модели
# правильные метки / предсказанные
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")
print("===========================================================")

# Вычисляем confusion matrix
# предсказано | не купил  |  купил
# ------------|-------------------------------
# не купил    | правильно |  ошибся
# купил       | ошибся    |  правильно
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("===========================================================")

# Вычисляем precision, recall, f1-score
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
print("===========================================================")

# Визуализация результатов на обучающей выборке
X_set, y_set = X_train.values, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Визуализация результатов на тестовой выборке
X_set, y_set = X_test.values, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()