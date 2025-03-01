import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

data1 = pd.read_csv('data.csv')

data1.dropna(inplace=True)

#cоздаем экземпляр MinMaxScaler и масштабируем все признаки
scaler = MinMaxScaler()
data = scaler.fit_transform(data1)

#применяем DBSCAN
dbscan = DBSCAN(eps=0.1, min_samples=2)  #параметры eps-расстояние на котором искать соседей , min_samples-количество требуемых соседей для создания кластера
dbscan.fit(data)

#выводим метки (номер кластера, к которому он принадлежит)
print(f'Clusters labels: {dbscan.labels_}')

#строим график
plt.figure(figsize=(6,6))
plt.title('DBSCAN Clustering')
plt.scatter(data[:, 0], data[:, 1], c=dbscan.labels_, cmap='rainbow')
plt.savefig('dbscan.png')
plt.show()

#оценка качества кластеризации с использованием silhouette_score
#DBSCAN помечает выбросы меткой -1, которые нужно исключить для оценки
labels = dbscan.labels_
if len(set(labels)) > 1 and -1 in labels:
    score = silhouette_score(data[labels != -1], labels[labels != -1])
    print(f'Silhouette Score (excluding noise): {score}')
else:
    print('Silhouette Score cannot be computed (only one cluster or all points are noise)')
