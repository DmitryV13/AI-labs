import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap

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


plt.scatter(X_train['Age'], X_train['Salary'], c=y_train, cmap=ListedColormap(('red', 'green')))
plt.xlabel('Age')
plt.ylabel('Salary')

# Визуализация результатов

def plot_lines(X_set, feature_name, rules):
    if feature_name == 'Age':  # Вертикальные линии для Age
        for value in rules.keys():
            plt.axvline(x=value, color='blue', linestyle='--', alpha=0.7)
    elif feature_name == 'Salary':  # Горизонтальные линии для Salary
        for value in rules.keys():
            plt.axhline(y=value, color='blue', linestyle='--', alpha=0.7)

# Визуализация результатов на тренировочной выборке
X_set, y_set = X_train.values, y_train
y_pred_train = one_r.predict(X_train)

cmap = ListedColormap(('red', 'green'))

plt.figure(figsize=(8, 6))

# Верно классифицированные точки
correct_train = y_pred_train == y_set
plt.scatter(X_set[correct_train, 0], X_set[correct_train, 1], c=y_set[correct_train], cmap=cmap, label="Correct", edgecolors='black', marker='o')

# Неверно классифицированные точки
incorrect_train = ~correct_train
plt.scatter(X_set[incorrect_train, 0], X_set[incorrect_train, 1], c=y_set[incorrect_train], cmap=cmap, label="Incorrect", edgecolors='black', marker='x')

# Рисуем линии для тренировочных данных
plot_lines(X_set, one_r.feature, one_r.rules)
plt.title('OneR Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Визуализация результатов на тестовой выборке
X_set, y_set = X_test.values, y_test
y_pred_test = one_r.predict(X_test)

plt.figure(figsize=(8, 6))

# Верно классифицированные точки
correct_test = y_pred_test == y_set
plt.scatter(X_set[correct_test, 0], X_set[correct_test, 1], c=y_set[correct_test], cmap=cmap, label="Correct", edgecolors='black', marker='o')

# Неверно классифицированные точки
incorrect_test = ~correct_test
plt.scatter(X_set[incorrect_test, 0], X_set[incorrect_test, 1], c=y_set[incorrect_test], cmap=cmap, label="Incorrect", edgecolors='black', marker='x')

# Рисуем линии для тестовых данных
plot_lines(X_set, one_r.feature, one_r.rules)
plt.title('OneR Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()