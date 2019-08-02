from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

df = load_iris()
x = df.data
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
res = svc.predict(x_test)

cm = confusion_matrix(res, y_test)
print(cm)
print("Accuracy = ", accuracy_score(res, y_test))

crossValScore = cross_val_score(svc, x_train, y_train, cv=3)
print("Cross Validation Score = ", crossValScore.mean())
print("Standard Deviation = ", crossValScore.std())

params = [{'C': [1, 2, 3, 4, 5], 'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [1, .5, .1, .01, .001]}]
gs = GridSearchCV(svc, params, 'accuracy', -1, cv=3, iid=True)
gridSearch = gs.fit(x_train, y_train)
print(gridSearch.best_score_)
print(gridSearch.best_params_)
