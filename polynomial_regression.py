import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('salaries.csv')
df = df.rename(index=str, columns={"unvan": "Position", "Egitim Seviyesi": "Education Level", "maas": "Salary"})
print(df)

edLevel = df.iloc[:, [1]].values
salary = df.iloc[:, [2]].values

plt.plot(edLevel, salary)
plt.xlabel("Level")
plt.ylabel("Salaries")
plt.show()

lReg = LinearRegression()
lReg.fit(edLevel, salary)
plt.scatter(edLevel, salary, color='red')
plt.plot(edLevel, lReg.predict(edLevel), color='blue')
plt.show()

pRegF = PolynomialFeatures(degree=2)
edLevelPoly = pRegF.fit_transform(edLevel)
print(edLevelPoly)
pReg = LinearRegression()
pReg.fit(edLevelPoly, salary)
plt.scatter(edLevel, salary, color='red')
plt.plot(edLevel, pReg.predict(edLevelPoly), color='blue')
plt.show()

pRegF = PolynomialFeatures(degree=4)
edLevelPoly = pRegF.fit_transform(edLevel)
print(edLevelPoly)
pReg.fit(edLevelPoly, salary)
plt.scatter(edLevel, salary, color='red')
plt.plot(edLevel, pReg.predict(edLevelPoly), color='blue')
plt.show()
