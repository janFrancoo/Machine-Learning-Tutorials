import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv("veriler.csv")

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

age = df.iloc[:, 1:4].values
age[:, 1:4] = imputer.fit_transform(age[:, 1:4])
print(age)
sc = StandardScaler()

age = sc.fit_transform(age)
print(age)

le = LabelEncoder()
ohe = OneHotEncoder()

country = df.iloc[:, 0:1].values
country[:, 0] = le.fit_transform(country[:, 0])
country = ohe.fit_transform(country).toarray()

result = pd.DataFrame(data=country, index=range(22), columns=['fr', 'tr', 'us'])
result_2 = pd.DataFrame(data=age, index=range(22), columns=['height', 'weight', 'age'])

gender = df.iloc[:, -1:].values
df = pd.concat([result, result_2], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df, gender, test_size=0.33, random_state=0)

print(x_train)
print(x_train.shape)
print(x_test)
print(x_test.shape)
print(y_train)
print(y_train.shape)
print(y_test)
print(y_test.shape)
