import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pandas import read_csv
from imblearn.under_sampling import RandomUnderSampler


print('tensorflow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)
print('\n\n\n')


#Exited 0 = остался клиентом банка, 1 = ушел

# Surname - Фамилия
# CreditScore - Кредитный рейтинг клиента
# Geography - Где проживает
# Gender - Пол
# Age - Возраст
# Tenure - Количество лет, в течение которых клиент работает с банком
# Balance - Остаток на банковском счете клиента
# NumOfProducts - Количество банковских продуктов, которыми пользуется клиент
# HasCrCard - Двоичный флаг, указывающий, есть ли у клиента кредитная карта в банке или нет
# IsActiveMember - Двоичный флаг, указывающий, является ли клиент активным членом банка или нет
# EstimatedSalary - Предполагаемая зарплата клиента в долларах
# Exited - Ушел клиент или остался (метка)


#################################################################
#Раздел 1: Исследовательский анализ данных
#################################################################

df = read_csv("Churn_Modelling.csv")

#выводим информацию о датасете для анализа
print(df.head())
print("===============================")
print(df.info())
print("===============================")

#выводим график значений метки и количества обьектов принадлежащих конкретному значению
plt.figure(figsize=(6, 4))
sns.countplot(x="Exited", hue="Exited", data=df, palette={0: "red", 1: "blue"}, legend=False)
plt.xlabel("Покинул/остался по отношению к банку")
plt.ylabel("Количество случаев")
plt.title("Статистика клиентов")
plt.show()

#матрица коррелляции
df_features = df.drop(columns=["Exited"])
corr_matrix = df_features.corr(numeric_only=True)

# Устанавливаем более широкую ширину отображения
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
print(corr_matrix)
print("===============================")

#график корреляции
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="cividis", fmt=".2f")
plt.title("Корреляционная матрица")
plt.show()

# Строим boxplot для CreditScore сгруппированный по Exited
# (изучаем распределение данных для признака CreditScore)
plt.figure(figsize=(10, 6))
sns.boxplot(x="Exited", y="CreditScore", data=df, hue="Exited", palette={0: "red", 1: "green"}, legend=False)
plt.title("Boxplot для признака CreditScore, разделенный по Exited")
plt.show()

#расчитываю сводную статистику для CreditScore, сгруппированную по Exited
grouped_summary = df.groupby('Exited')['CreditScore'].describe()
print(grouped_summary)
print("===============================")

#выводим уникальные значения для столбца Geography
value_counts = df["Geography"].value_counts()
print(value_counts)
print("===============================")

#Создайте countplot для Geography. Установите параметр hue равным Exited.
plt.figure(figsize=(10, 6))
sns.countplot(x="Geography", hue="Exited", data=df)
plt.xlabel("Страна")
plt.ylabel("Количество")
plt.title("Распределение клиентов по странам и метке Exited")
plt.legend(title="Exited", labels=["Остались", "Ушли"])
plt.show()

#Создайте countplot для Age. Установите параметр hue равным Exited.
plt.figure(figsize=(20, 6))
sns.countplot(x="Age", hue="Exited", data=df)
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.title("Распределение клиентов по возрасту и метке Exited")
plt.legend(title="Exited", labels=["Остались", "Ушли"])
plt.show()



# Группируем данные по Age и Exited
age_counts = df.groupby(["Age", "Exited"]).size().unstack(fill_value=0)

# Вычисляем общую сумму для каждого возраста
total_counts = age_counts.sum(axis=1)

# Доля ушедших клиентов
exit_ratio = age_counts[1] / total_counts

# Исходные цвета (в формате RGB [0-1])
palette = sns.color_palette("deep", 2)
color_0 = np.array(palette[0])  # Первый цвет
color_1 = np.array(palette[1])  # Второй цвет

# Смешивание цветов
mixed_colors = [(1 - r) * color_0 + r * color_1 for r in exit_ratio]

# Преобразуем цвета в формат hex для Matplotlib
hex_colors = ["#" + "".join(f"{int(c*255):02x}" for c in color) for color in mixed_colors]

# Строим график
plt.figure(figsize=(20, 6))
bars = plt.bar(age_counts.index, total_counts, color=hex_colors)
plt.xticks(age_counts.index)
plt.xlim(age_counts.index[0] - 0.5, age_counts.index[-1] + 0.5)
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.title("Распределение клиентов по возрасту (смешанный цвет Exited)")
plt.show()

#в возрастном диапазоне 49-59 клиенты часто покидают банк, изолтруем этот диапазон
# Фильтруем DataFrame по диапазону значений Age
df_filtered = df[(df['Age'] >= 49) & (df['Age'] <= 59)]

# Строим график для отфильтрованных данных
plt.figure(figsize=(20, 6))
sns.countplot(x="Age", hue="Exited", data=df_filtered)
plt.xlabel("Возраст")
plt.ylabel("Количество")
plt.title("Распределение клиентов по возрасту и метке Exited (20-50 лет)")
plt.legend(title="Exited", labels=["Остались", "Ушли"])
plt.show()


#################################################################
#Раздел 2: Исследовательский анализ данных
#################################################################
print('\n\n\n')

#датасет содержит 10000 строк

# Создание Series с количеством пропущенных значений в каждом столбце
missing_values = df.isnull().sum()
print(missing_values)
print("===============================")

# Преобразуем количество пропущенных значений в проценты
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)
print("===============================")

value_counts = df["CreditScore"].value_counts()
print(value_counts)
print("===============================")
# Credit Score является одним из важных признаков поэтому его удалять нельзя,
# далее пропущенные значения будут заполнены

value_counts = df["HasCrCard"].value_counts()
print(value_counts)
print("===============================")

value_counts = df["Gender"].value_counts()
print(value_counts)
print("===============================")
# Gender, HasCrCard являются категориальным и числовым признаками содержащими
# по 2 возможных значения, их можно не удалять
#Сколько уникальных названий рабочих должностей существует?


#countplot для HasCrCard.
plt.figure(figsize=(10, 6))
sns.countplot(x="HasCrCard", hue="HasCrCard", data=df)
plt.xlabel("HasCrCard")
plt.ylabel("Количество")
plt.title("Распределение клиентов по HasCrCard")
plt.show()

#countplot для HasCrCard. Установите параметр hue равным Exited.
plt.figure(figsize=(10, 6))
sns.countplot(x="HasCrCard", hue="Exited", data=df)
plt.xlabel("HasCrCard")
plt.ylabel("Количество")
plt.title("Распределение клиентов по HasCrCard и метке Exited")
plt.show()
# Этот график не дает нам полной информации о том, существует ли сильная связь между
# HasCrCard и Exited. Нам необходимо определить процент ушедших клиентов в каждой категории


# Группируем по 'HasCrCard' и вычисляем процент ушедших клиентов в каждой группе
print("% of clients who exited")
exited_percentage = df.groupby('HasCrCard')['Exited'].mean() * 100
print(exited_percentage)
print("===============================")

# Строим bar plot для отображения процента ушедших клиентов
plt.figure(figsize=(10, 6))
sns.barplot(x=exited_percentage.index, y=exited_percentage.values)
plt.xlabel("HasCrCard")
plt.ylabel("Процент ушедших клиентов")
plt.title("Процент ушедших клиентов по признаку HasCrCard")
plt.show()
# процент ушедших клиентов почти одинаков, а значит мы можем удалить признак HasCrCard из нашего датасета

# Удаление столбца HasCrCard
df = df.drop("HasCrCard", axis=1)

# проверяем количество пропущенных значений для каждого признака
print("Пропущенные значения после удаления HasCrCard")
missing_values = df.isnull().sum()
print(missing_values)
print("===============================")

# преобразование категориальных признаков столбца Gender
category_mapping = {"Female": 0, "Male": 1}
df["Gender"] = df["Gender"].map(category_mapping)

# Заменить пустое значение Gender на среднее значение столбца
df["Gender"] = df["Gender"].fillna(df["Gender"].mean())

# удаление пустых значений CreditScore так как их доля меньше 0.1%
df = df.dropna(subset=["CreditScore"])

# проверяем количество пропущенных значений для каждого признака
print("Пропущенные значения после обработки признаков CreditScore и Gender")
missing_values = df.isnull().sum()
print(missing_values)
print("===============================")


# Выбираем все оставшиеся столбцы, которые не являются числовыми
print("Нечисловые признаки")
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
print(non_numeric_columns)
print("===============================")

#проверяем количество уникальных значений в столбце Surname
value_counts = df["Surname"].value_counts()
print(value_counts)
print("===============================")
#слишком много уникальных значений, удаляем столбец

# Удаление столбца Surname
df = df.drop("Surname", axis=1)

# обрабатываем столбец Geography
# Создание экземпляра LabelEncoder
encoder = LabelEncoder()

# Преобразование категориального признака Geography в числовой
df["Geography"] = encoder.fit_transform(df["Geography"])

#################################################################
# Выборка данныхи их нормализация
#################################################################
# выравниваем количество обьектов для каждого значения метки
# Разделим данные на два подмножества по метке Exited
df_0 = df[df['Exited'] == 0]  # строки, где Exited == 0
df_1 = df[df['Exited'] == 1]  # строки, где Exited == 1

# Определим, сколько строк нужно оставить для каждой метки
df_0_sampled = df_0.sample(n=len(df_1), random_state=42)  # случайный выбор строк из df_0

# Объединяем выборки, чтобы создать сбалансированный датасет
df_balanced = pd.concat([df_0_sampled, df_1])

#################################################################

x = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary']]
y = df['Exited']

#создаем набор данных для обучения и тестирования
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Применяем under-sampling на обучающих данных
rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
x_train_resampled, y_train_resampled = rus.fit_resample(x_train, y_train)

# Инициализируем скейлер
scaler = MinMaxScaler()

# Подгонка скейлера только на x_train и x_test раздельно для предотвращения утечки
X_train_scaled = scaler.fit_transform(x_train_resampled)
X_train_scaled1 = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


# Проверяем размер классов после балансировки
print("До балансировки")
print(y_train.value_counts())  # До under-sampling
print("===============================")

print("После балансировки")
print(pd.Series(y_train_resampled).value_counts())  # После under-sampling
print("===============================")

#################################################################
#Раздел 3: Создание модели
#################################################################
print('\n\n\n')

#создание модели
model = Sequential()
model.add(Dense(78, activation='relu', input_shape=(x_train_resampled.shape[1],))) # x_train
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Рассчитываем веса классов
# class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
# class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

#обучение модели
#model.fit(x_train, y_train, epochs=25, batch_size=256, validation_data=(x_test, y_test))
model.fit(x_train_resampled, y_train_resampled, epochs=25, batch_size=256, validation_data=(x_test, y_test))

#сохранение модели
model.save('bank_model.keras')

#визуализация потерь
pd.DataFrame(model.history.history)[['loss', 'val_loss']].plot()
plt.title('Потери модели на тренировочном и валидационном наборах')
plt.xlabel('Эпохи')
plt.ylabel('Значение потерь')
plt.show()

#Создаем предсказания для тестового набора
predictions = (model.predict(x_test) > 0.5).astype(int)

#отчет о классификации
print('Classification report')
print(classification_report(y_test, predictions))
print("===============================")

#матрица ошибок
print('Confusion matrix')
print(confusion_matrix(y_test, predictions))
print("===============================")

#проверяем клиента на уход
# < 50% клиент останется
# > 50% клиент уйдет
new_data = pd.DataFrame([[581,'Germany','Female',32,1,103633.04,1,0,110431.51]],
                        columns=['CreditScore','Geography','Gender','Age','Tenure','Balance',
                                 'NumOfProducts','IsActiveMember','EstimatedSalary'])

new_data["Geography"] = encoder.fit_transform(new_data["Geography"])
new_data["Gender"] = new_data["Gender"].map(category_mapping)
test_line = scaler.transform(new_data)

print(model.predict(test_line))