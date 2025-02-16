import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap

# Загружаем датасет
dataset = pd.read_csv('gnb1.csv')
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

# Реализация OneR Classifier
class OneRClassifier:
    def __init__(self):
        self.rules = {}
        self.feature = None

# Вычисление лучшего признака
    def fit(self, X, y):
        best_feature = None
        best_accuracy = -1
        best_rule = None

        for column in X.columns:
            rule = self._create_rule(X[column], y)
            accuracy = self._rule_accuracy(rule, X[column], y)

            if accuracy > best_accuracy:
                best_feature = column
                best_accuracy = accuracy
                best_rule = rule

        self.feature = best_feature
        self.rules = best_rule

    # если значение отсутствует возвращаем наиболее частую этикетку
    def predict(self, X):
        if self.feature not in X.columns:
            raise ValueError(f"Feature '{self.feature}' not found in input data")

        most_common_label = max(set(self.rules.values()), key=list(self.rules.values()).count)

        return X[self.feature].apply(lambda x: self.rules.get(x, most_common_label))

    # Создание правила
    def _create_rule(self, feature, target):
        rule = {}
        for value in feature.unique():
            mask = feature == value
            target_value = target[mask].mode()[0]  # Берем наиболее частое значение
            rule[value] = target_value
        return rule

    def _rule_accuracy(self, rule, feature, target):
        predictions = feature.apply(lambda x: rule.get(x, 0))
        return accuracy_score(target, predictions)

# Обучаем модель OneR
one_r = OneRClassifier()
one_r.fit(X_train, y_train)

# Делаем предсказания
y_pred = one_r.predict(X_test)

# Предсказание для человека 62 лет с зарплатой 20000 на покупку телефона
print("===========================================================")
print("Купит ли человек 62 лет с зарплатой 20000?")
new_data = pd.DataFrame([[62, 20000]], columns=['Age', 'Salary'])
print(one_r.predict(new_data).values[0])
print("===========================================================")

# Выводим точность модели
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


X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set['Age'].min() - 10, stop = X_set['Age'].max() + 10, step = 0.25),
                     np.arange(start = X_set['Salary'].min() - 1000, stop = X_set['Salary'].max() + 1000, step = 0.25))

# Получаем предсказания для сетки
grid_points = np.array([X1.ravel(), X2.ravel()]).T
predictions = one_r.predict(pd.DataFrame(grid_points, columns=['Age', 'Salary']))

# Отображаем контуры решения
plt.contourf(X1, X2, np.array(predictions).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Отображаем обучающие точки
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set['Age'][y_set == j], X_set['Salary'][y_set == j], c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('OneR (Training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Визуализация результатов на тестовом наборе данных
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set['Age'].min() - 10, stop = X_set['Age'].max() + 10, step = 0.25),
                     np.arange(start = X_set['Salary'].min() - 1000, stop = X_set['Salary'].max() + 1000, step = 0.25))

# Получаем предсказания для сетки
grid_points = np.array([X1.ravel(), X2.ravel()]).T
predictions = one_r.predict(pd.DataFrame(grid_points, columns=['Age', 'Salary']))

# Отображаем контуры решения
plt.contourf(X1, X2, np.array(predictions).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Отображаем тестовые точки
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set['Age'][y_set == j], X_set['Salary'][y_set == j], c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('OneR (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()