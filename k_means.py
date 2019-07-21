import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('musteriler.csv')
print(df)
x = df.iloc[:, [3, 4]]

res = []
for i in range(1, 10):
    kMeans = KMeans(n_clusters=i)
    kMeans.fit(x)
    res.append(kMeans.inertia_)

plt.plot(range(1, 10), res)
plt.show()

kMeans = KMeans(n_clusters=3)
kMeans.fit(x)
pl.figure('Before Clustering')
pl.scatter(x.values[:, 0], x.values[:, 1])
pl.show()
pl.figure('After Clustering')
pl.scatter(x.values[:, 0], x.values[:, 1], c=kMeans.labels_)
pl.show()
