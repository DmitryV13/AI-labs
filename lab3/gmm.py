import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

data1 = pd.read_csv('customers.csv')

#создаем экземпляр MinMaxScaler и масштабируем все признаки
scaler = MinMaxScaler()
data = scaler.fit_transform(data1)

#применяем Gaussian Mixture Model
gmm = GaussianMixture(n_components=3)
gmm.fit(data)

#выводим центры всех кластеров
print(f'Clusters centers cords: {gmm.means_}')

#выводим метки (номер кластера, к которому он принадлежит) для всех объектов
labels = gmm.predict(data)
print(f'Clusters features labels: {labels}')

#строим график
plt.figure(figsize=(6,6))
plt.title('Gaussian Mixture Model (GMM)')
plt.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')
plt.savefig('gmm.png')
plt.show()

#оценка качества кластеризации с использованием silhouette_score
score = silhouette_score(data, labels)
print(f'Silhouette Score: {score}')
