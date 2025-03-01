import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data1 = pd.read_csv('customers.csv')

#создаем экземпляр MinMaxScaler и масштабируем все признаки
scaler = MinMaxScaler()
data = scaler.fit_transform(data1)

kmeans = KMeans(n_clusters=6)
kmeans.fit(data)

#выводим центры всех кластеров
print(f'Clusters centers cords: {kmeans.cluster_centers_}')

#выводим метки(номер кластера, к которому он принадлежит) для всех обьектов
print(f'Clusters features labels: {kmeans.labels_}')

#строим график
plt.figure(figsize=(6,6))
plt.title('K Means')
plt.scatter(data[:,0], data[:,1], c=kmeans.labels_, cmap='rainbow')
plt.savefig('kmeans.png')
plt.show()

#оценка качества кластеризации с использованием silhouette_score
score = silhouette_score(data, kmeans.labels_)
print(f'Silhouette Score: {score}')