import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('Churn_Modelling.csv')
print(df)

le = LabelEncoder()
ohe = OneHotEncoder()
df.iloc[:, [5]] = le.fit_transform(df.iloc[:, [5]])
geography = df.iloc[:, [4]].values
geography = ohe.fit_transform(geography).toarray()
result = pd.DataFrame(data=geography, index=range(10000), columns=['fr', 'sp', 'ge'])

sc = StandardScaler()
mustScale = df.iloc[:, [3, 6, 7, 8, 9, 12]].values
mustScale = sc.fit_transform(mustScale)
result2 = pd.DataFrame(data=mustScale, index=range(10000), columns=['creditScore', 'age', 'tenure', 'balance',
                                                                    'numOfProducts', 'estimatedSalary'])

CardActive = df.iloc[:, [10, 11]]
x = pd.concat([result, result2, CardActive], axis=1)
y = df.iloc[:, [13]].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=0)
print(x_train)
print(y_train)

model = Sequential()
model.add(Dense(6, activation='relu', input_dim=11))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50)
res = model.predict(x_test)
res = (res > .5)
cm = confusion_matrix(res, y_test)
print(cm)
