from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

df = load_iris()
x = df.data
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
res = knn.predict(x_test)

cm = confusion_matrix(res, y_test)
print(cm)
print("Accuracy = ", accuracy_score(res, y_test))

crossValScore = cross_val_score(knn, x_train, y_train, cv=3)
print("Cross Validation Score = ", crossValScore.mean())
print("Standard Deviation = ", crossValScore.std())
