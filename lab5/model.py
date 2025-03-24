import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from tqdm import tqdm
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import (Dense, Dropout, Activation, MaxPooling2D, Conv2D, Flatten,
                                          BatchNormalization, GlobalMaxPool2D)

# вывод количества изображений в каждой категории
print("Number of horses images")
print(len(os.listdir("/kaggle/input/lab-work-5/horses")))
print("=========================================")

print("Number of dogs images")
print(len(os.listdir("/kaggle/input/lab-work-5/dogs")))
print("=========================================")

# получение списка всех путей к изображениям
DataPath = pathlib.Path("/kaggle/input/lab-work-5")
all_paths = DataPath.glob("*/*.jpg")

all_paths=list(all_paths)
print(all_paths[:10])
print("=========================================")

# преобразование путей в строки
all_paths=list(map(lambda x: str(x), all_paths))
for path in all_paths[:10]:
    print(path)
print("=========================================")

# перемешивание списка путей
shuffle(all_paths)
shuffle(all_paths)

for path in all_paths[:10]:
    print(path)
print("=========================================")

# проверка качества изображений
def TestImageQuality(all_paths):
    new_all_paths = []
    for path in tqdm(all_paths):
        try:
            image = tf.io.read_file(path)
            image = tf.io.decode_jpeg(image, channels=3)
        except:
            continue
        new_all_paths.append(path)
    return new_all_paths

all_paths = TestImageQuality(all_paths)

for path in all_paths[:10]:
    print(path)
print("=========================================")

# получение метки класса из пути файла
def get_label(image_path):
    return image_path.split("/")[-2]

# создание списка меток классов
all_labels = list(map(lambda x: get_label(x), all_paths))
for label in all_labels[:10]:
    print(label)
print("=========================================")

# преобразование текстовых меток в числовые
le = LabelEncoder()
all_labels = le.fit_transform(all_labels)
for label in all_labels[:10]:
    print(label)
print("=========================================")

# разбиение данных на обучающую и валидационную выборки
Train_paths, Val_paths, Train_labels, Val_labels = train_test_split(all_paths, all_labels)
print(Train_paths[:10], Train_labels[:10])

# функция загрузки изображения и метки
def load(image, label):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    return image, label

IMG_SIZE = 224
BATCH_SIZE = 128

# изменение размера изображений
resize = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE)
])

# аугментация данных
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(height_factor=(-0.3, -0.2))
])

#оптимизация загрузки данных в процессе обучения
AUTOTUNE = tf.data.experimental.AUTOTUNE

# создание набора данных
def get_dataset(paths, labels, train=True):
    image_paths = tf.convert_to_tensor(paths)
    labels = tf.convert_to_tensor(labels)

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    dataset = dataset.map(lambda image, label: load(image, label))
    dataset = dataset.map(lambda image, label: (resize(image), label), num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)

    if train:
        dataset = dataset.map(lambda image, label: (data_augmentation(image), label), num_parallel_calls=AUTOTUNE)
        dataset = dataset.repeat()

    return dataset

# создание обучающего набора данных
train_dataset = get_dataset(Train_paths, Train_labels)

image, label = next(iter(train_dataset))
print(image.shape)
print(label.shape)
print("=========================================")

# вывод первой метки в обратном преобразовании для набора обучения
print(le.inverse_transform(label)[0])
print("=========================================")
plt.imshow((image[0].numpy()/255).reshape(224, 224, 3))
plt.show()

# создание валидационного набора данных
val_dataset = get_dataset(Val_paths, Val_labels)

image, label = next(iter(val_dataset))
print(image.shape)
print(label.shape)
print("=========================================")

# вывод первой метки в обратном преобразовании для набора тестирования
print(le.inverse_transform(label)[0])
print("=========================================")
plt.imshow((image[0].numpy()/255).reshape(224, 224, 3))
plt.show()

# создание модели
model = tf.keras.Sequential()

# первый сверточный слой
model.add(Conv2D(input_shape=(224, 224, 3), padding='same', filters=32, kernel_size=(7, 7)))
model.add(Activation('relu'))
model.add(BatchNormalization()) #нормализация батчей для улучшения обучения
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# второй сверточный слой
model.add(Conv2D(padding='valid', filters=64, kernel_size=(5, 5)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) #пулинг с размером 2x2 для уменьшения размерности данных
model.add(Dropout(0.2))

# третий сверточный слой
model.add(Conv2D(padding='valid', filters=128, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

# четвертый сверточный слой
model.add(Conv2D(padding='valid', filters=256, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

# пятый сверточный слой
model.add(Conv2D(filters=256, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# глобальный пулинг
model.add(GlobalMaxPool2D()) #сокращает размерность данных до одного значения для каждого канала

# полносвязный слой
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# выходной слой
model.add(Dense(1))
model.add(Activation('sigmoid'))

print(model.summary())
print("=========================================")

# компиляция модели
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# обучение модели
history = model.fit(
    train_dataset,
    steps_per_epoch=len(Train_paths)//BATCH_SIZE,
    epochs=10,
    validation_data=val_dataset,
    validation_steps=len(Val_paths)//BATCH_SIZE
)

# оценка модели
loss, accuracy = model.evaluate(val_dataset)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

