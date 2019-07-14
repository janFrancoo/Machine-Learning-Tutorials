import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('salaries.csv')
df = df.rename(index=str, columns={"unvan": "Position", "Egitim Seviyesi": "Education Level", "maas": "Salary"})
print(df)

scaler = StandardScaler()
level = scaler.fit_transform(df.iloc[:, [1]].values)
salary = scaler.fit_transform(df.iloc[:, [2]].values)

supportVectorReg = SVR(kernel='rbf')
supportVectorReg.fit(level, salary)

plt.scatter(level, salary)
plt.plot(level, supportVectorReg.predict(level))
plt.show()

supportVectorReg = SVR(kernel='linear')
supportVectorReg.fit(level, salary)

plt.scatter(level, salary)
plt.plot(level, supportVectorReg.predict(level))
plt.show()

supportVectorReg = SVR(kernel='poly')
supportVectorReg.fit(level, salary)

plt.scatter(level, salary)
plt.plot(level, supportVectorReg.predict(level))
plt.show()
