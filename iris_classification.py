from sklearn.svm import SVC
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = datasets.load_iris()
print(df)

x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=.33, random_state=0)

logisticReg = LogisticRegression(multi_class='auto')
logisticReg.fit(x_train, y_train)
logisticRegRes = logisticReg.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=logisticRegRes)
print("***** Confusion Matrix - LogisticReg *****")
print(confusionMatrix)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knnRes = knn.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=knnRes)
print("***** Confusion Matrix - KNN *****")
print(confusionMatrix)

supportVector = SVC(kernel='rbf', gamma='auto')
supportVector.fit(x_train, y_train)
supportVectorRes = supportVector.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=supportVectorRes)
print("***** Confusion Matrix - SVC *****")
print(confusionMatrix)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
treeRes = tree.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=treeRes)
print("***** Confusion Matrix - Decision Tree *****")
print(confusionMatrix)

randomForest = RandomForestClassifier(n_estimators=10)
randomForest.fit(x_train, y_train)
randomForestRes = randomForest.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=randomForestRes)
print("***** Confusion Matrix - Random Forest *****")
print(confusionMatrix)

gaussianNB = GaussianNB()
gaussianNB.fit(x_train, y_train)
gaussianNBRes = gaussianNB.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=gaussianNBRes)
print("***** Confusion Matrix - Gaussian Naive Bayes *****")
print(confusionMatrix)
