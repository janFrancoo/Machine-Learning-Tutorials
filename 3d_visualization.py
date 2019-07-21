from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = datasets.load_iris()
print(df)

plt.figure(2, figsize=(7, 4))
plt.scatter(df.data[:, [0, 1]][:, 0], df.data[:, [0, 1]][:, 1], c=df.target, cmap=plt.cm.coolwarm)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig)
ax.scatter(df.data[:, [0]], df.data[:, [1]], df.data[:, [2]], c=df.target, cmap=plt.cm.coolwarm)
plt.show()
