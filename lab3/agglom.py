import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

#загружаем данные
data1 = pd.read_csv('customers.csv')

#создаем экземпляр MinMaxScaler и масштабируем все признаки
scaler = MinMaxScaler()
data = scaler.fit_transform(data1)

#применяем агломеративную кластеризацию
agglomerative = AgglomerativeClustering(distance_threshold=1, n_clusters=None)
agglomerative.fit(data)

#выводим метки (номер кластера, к которому объект принадлежит)
print(f'Clusters features labels: {agglomerative.labels_}')

#строим график
plt.figure(figsize=(6, 6))
plt.title('Agglomerative Clustering')
plt.scatter(data[:, 0], data[:, 1], c=agglomerative.labels_, cmap='rainbow')
plt.savefig('agglom.png')
plt.show()

#строим матрицу расстояний для дендрограммы
linked = linkage(data, method='ward')

#визуализация дендрограммы
plt.figure(figsize=(10, 7))
dendrogram(linked, color_threshold=1)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')
plt.show()

#оценка качества кластеризации с использованием silhouette_score
score = silhouette_score(data, agglomerative.labels_)
print(f'Silhouette Score: {score}')

