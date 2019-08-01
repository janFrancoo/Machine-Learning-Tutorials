import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('Wine.csv')
print(df)

x = df.iloc[:, 0:13].values
y = df.iloc[:, 13].values

sc = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=0)
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

lda = LinearDiscriminantAnalysis(n_components=2)
x_train2 = lda.fit_transform(x_train, y_train)
x_test2 = lda.transform(x_test)

logisticReg = LogisticRegression(random_state=0)
logisticReg.fit(x_train, y_train)
res = logisticReg.predict(x_test)

logisticReg2 = LogisticRegression(random_state=0)
logisticReg2.fit(x_train2, y_train)
res2 = logisticReg2.predict(x_test2)

cm1 = confusion_matrix(res, y_test)
cm2 = confusion_matrix(res2, y_test)
print(cm1)
print(cm2)
