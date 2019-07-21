import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = datasets.load_iris()
print(df.data)
print(df.target)

attrNames = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
names = ["Logistic Reg", "K-NN", "SVM RBF", "Decision Tree", "Random Forest", "Naive Bayes"]
classifiers = [LogisticRegression(multi_class='auto'), KNeighborsClassifier(n_neighbors=5), SVC(gamma='auto'),
               DecisionTreeClassifier(), RandomForestClassifier(n_estimators=10)]

fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig)
ax.scatter(df.data[:, [0]], df.data[:, [1]], df.data[:, [2]], c=df.target, cmap=ListedColormap(['#fff700',
                                                                                                '#524545', '#0af7ef']))
plt.show()

for i, j in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
    data = df.data[:, [i, j]]
    target = df.target
    xMin, xMax = data[:, 0].min() - .5, data[:, 0].max() + .5
    yMin, yMax = data[:, 1].min() - .5, data[:, 1].max() + .5
    xRange, yRange = np.meshgrid(np.arange(xMin, xMax, .02), np.arange(yMin, yMax, .02))
    plt.figure(figsize=(7, 4))
    plt.scatter(data[:, 0], data[:, 1], c=df.target, cmap=ListedColormap(['#fff700', '#524545', '#0af7ef']))
    plt.xlabel(attrNames[i])
    plt.ylabel(attrNames[j])
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.33, random_state=0)
    k = 1
    plt.figure(figsize=(15, 4))
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(1, len(classifiers), k)
        clf.fit(x_train, y_train)
        res = clf.predict(np.c_[xRange.ravel(), yRange.ravel()])
        res = res.reshape(xRange.shape)
        ax.contourf(xRange, yRange, res, cmap=ListedColormap(['#c91414', '#2914c9', '#a10e83']), alpha=.8)
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=ListedColormap(['#fff700', '#524545', '#0af7ef']),
                    edgecolors='k')
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=ListedColormap(['#fff700', '#524545', '#0af7ef']),
                    edgecolors='k', alpha=0.6)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        k += 1

    plt.tight_layout()
    plt.show()
