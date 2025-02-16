import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загружаем датасет
dataset = pd.read_csv('gnb1.csv')
X = dataset[['Age', 'Salary']]
y = dataset['Bought Iphone 14']

# Разбиваем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
# не купил - правильно, купил -ошибся
# купил - правильно, не купил -ошибся
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("===========================================================")

# Вычисляем precision, recall, f1-score
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
print("===========================================================")
