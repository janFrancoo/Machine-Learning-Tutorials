import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('veriler.csv')
df = df.rename(index=str, columns={"ulke": "country", "boy": "height",
                                   "kilo": "weight", "yas": "age", "cinsiyet": "gender"})

one_hot_encoder = OneHotEncoder()
country = df.iloc[:, [0]].values
country = one_hot_encoder.fit_transform(country).toarray()
country = pd.DataFrame(data=country, columns=['fr', 'tr', 'us'])

weight = df.iloc[:, [2]].values
weight = pd.DataFrame(data=weight, columns=['weight'])

imputer = SimpleImputer()
age = df.iloc[:, [3]].values
age = imputer.fit_transform(age)
age = pd.DataFrame(data=age, columns=['age'])

label_encoder = LabelEncoder()
gender = df.iloc[:, [4]].values
gender = label_encoder.fit_transform(gender[:, 0])
gender = pd.DataFrame(data=gender, columns=['gender'])

height = df[['height']]

df = pd.concat([country, weight, age, gender], axis=1)
print(df)

x_train, x_test, y_train, y_test = train_test_split(df, height, test_size=0.33, random_state=0)
print(x_train, x_train.shape)
print(y_train, y_train.shape)
print(x_test, x_test.shape)
print(y_test, y_test.shape)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print(y_predict)

x_l = df.iloc[:, [0, 1, 2, 3, 4, 5]].values
r_l = sm.OLS(height, x_l).fit()
print(r_l.summary())

x_train = x_train.iloc[:, [0, 1, 2, 3, 5]]
x_test = x_test.iloc[:, [0, 1, 2, 3, 5]]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print(y_predict)

x_train = x_train.iloc[:, [0, 1, 2, 3]]
x_test = x_test.iloc[:, [0, 1, 2, 3]]

reg = LinearRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print(y_predict)
