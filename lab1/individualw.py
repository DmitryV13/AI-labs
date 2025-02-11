import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Шаг 1
#Считываем файл data.csv в DataFrame под названием cycle
#Файл содержит данные о оценках, количестве сна, посещаемости учеников, часов учебы и
#социальноэкономической оценки

pds.set_option('display.max_columns', None)  #показываем все столбцы
pds.set_option('display.width', 1000)  #увеличиваем ширину вывода

cycle = pds.read_csv("data.csv")

#выводим первые строки (head) данных о учениках и получите информацию о них с
#помощью функций info() и describe()
print(cycle.head())
print("===============================")
print(cycle.info())
print("===============================")
print(cycle.describe())
print("===============================")

#Шаг 3
#проверяем корреляцию для колонок Study Hours и Grades
sns.jointplot(x='Study Hours', y='Grades', data=cycle, kind='scatter')
plt.show()

#Шаг 4
#проверяем корреляцию для колонок Study Hours и Attendance (%)
sns.jointplot(x='Study Hours', y='Attendance (%)', data=cycle, kind='scatter')
plt.show()

#Шаг 5
#сравниваем Attendance (%) и Grades
sns.jointplot(x='Attendance (%)', y='Grades', data=cycle, kind='hex')
plt.show()

#Шаг 6
#исследуем взаимосвязи во всем наборе данных
sns.pairplot(cycle[['Socioeconomic Score',
                        'Study Hours',
                        'Sleep Hours',
                        'Attendance (%)',
                        'Grades']], diag_kws={'bins': 10})
plt.show()

#Шаг 7
#создаем линейную модель графически
sns.lmplot(x='Study Hours', y='Grades', data=cycle)
plt.show()

#Шаг 8
#Определяем переменную X, содержащую числовые характеристики учеников, и
#переменную y, содержащую колонку Grades
X = cycle[['Study Hours', 'Attendance (%)', 'Sleep Hours', 'Socioeconomic Score']]
y = cycle['Grades']

#Шаг 9
#создаем набор данных для обучения и тестирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Шаги 10, 11, 12
#обучаем модель на наборе данных для обучения
lm = LinearRegression()
lm.fit(X_train, y_train)

#Шаг 13
#коэффициенты коэффициенты линейной регрессии
print("Model coefficients:")
print(f"Study Hours: {lm.coef_[0]}")
print(f"Attendance (%): {lm.coef_[1]}")
print(f"Sleep Hours: {lm.coef_[2]}")
print(f"Socioeconomic Score: {lm.coef_[3]}")
print("===============================")

#Шаг 14
#предсказания значений для X_test
predictions = lm.predict(X_test)

#Шаг 15
#построили диаграмму рассеяния реальных значений теста против предсказанных
#значений
plt.scatter(y_test, predictions)
plt.xlabel("Y Test")
plt.ylabel("Predicated Y")
plt.title("Scatter Plot")
plt.show()

#Шаг 16
#вычисляем среднюю абсолютную ошибку (MAE), среднеквадратичную ошибку (MSE),
#корень среднеквадратичной ошибки (RMSE) и коэффициент детерминации
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

#выводим
print(f"R^2: {r2}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

#Шаг 17
#вычисляем остатки
residuals = y_test - predictions

#строим для них гистограмму, чтобы увидеть нормальное распределение
sns.histplot(residuals, bins=50, kde=True)
plt.xlim(-50, 50)
plt.xlabel("Grades")
plt.show()