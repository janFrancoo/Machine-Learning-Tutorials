import pandas as pd
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv('veriler.csv')

missing = SimpleImputer()
age = df.iloc[:, [3]].values
age = missing.fit_transform(age)
age = pd.DataFrame(data=age, columns=['age'])

labelEncoder = LabelEncoder()
gender = df.iloc[:, [4]].values
gender = labelEncoder.fit_transform(gender.ravel())
gender = pd.DataFrame(data=gender, columns=['gender'])

oneHotEncoder = OneHotEncoder()
country = df.iloc[:, [0]].values
country = oneHotEncoder.fit_transform(country).toarray()
country = pd.DataFrame(data=country, columns=['fr', 'tr', 'us'])

hw = df.iloc[:, [1, 2]].values
hw = pd.DataFrame(data=hw, columns=['height', 'weight'])

df = pd.concat([country, hw, age, gender], axis=1)
print(df)

x = df.iloc[:, [3, 4, 5]]
y = df.iloc[:, [6]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

svr = SVC(kernel='rbf', gamma='auto')
svr.fit(x_train, y_train.values.ravel())
svrRes = svr.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=svrRes)
print("***** Confusion Matrix - RBF *****")
print("     E   K")
print("E   ", confusionMatrix[0][0], " ", confusionMatrix[0][1])
print("K   ", confusionMatrix[1][0], " ", confusionMatrix[1][1])

svr = SVC(kernel='linear', gamma='auto')
svr.fit(x_train, y_train.values.ravel())
svrRes = svr.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=svrRes)
print("***** Confusion Matrix - Linear *****")
print("     E   K")
print("E   ", confusionMatrix[0][0], " ", confusionMatrix[0][1])
print("K   ", confusionMatrix[1][0], " ", confusionMatrix[1][1])

svr = SVC(kernel='poly', gamma='auto')
svr.fit(x_train, y_train.values.ravel())
svrRes = svr.predict(x_test)

confusionMatrix = confusion_matrix(y_true=y_test, y_pred=svrRes)
print("***** Confusion Matrix - Poly *****")
print("     E   K")
print("E   ", confusionMatrix[0][0], " ", confusionMatrix[0][1])
print("K   ", confusionMatrix[1][0], " ", confusionMatrix[1][1])
