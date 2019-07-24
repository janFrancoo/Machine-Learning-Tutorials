import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('musteriler.csv')
x = df.iloc[:, [3, 4]].values
plt.scatter(x[:, 0], x[:, 1])
plt.show()
print(x)

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
res = hc.fit_predict(x)
print(res)

plt.scatter(x[:, 0], x[:, 1], c=res)
plt.show()

dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()
