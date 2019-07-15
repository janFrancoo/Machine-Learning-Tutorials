import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('salaries_new.csv')
print(df)
print(df.corr())

x = degSenScore = df.iloc[:, [2, 3, 4]].values
y = salary = df.iloc[:, [5]].values

linReg = LinearRegression()
linReg.fit(x, y)
linRegRes = linReg.predict(x)
model = sm.OLS(linReg.predict(x), x)
print(model.fit().summary())
print("***** RESULTS | R2 Scores *****")
print("Multiple Linear Regression - ", r2_score(y, linRegRes))
plt.title('Multiple Linear Regression')
plt.scatter(y, linRegRes, color='red')
plt.plot(y, y)
plt.show()

polyF = PolynomialFeatures(degree=4)
polyDeg = polyF.fit_transform(df.iloc[:, [2]].values)
polySen = polyF.fit_transform(df.iloc[:, [3]].values)
polyScore = polyF.fit_transform(df.iloc[:, [4]].values)
polyDeg = pd.DataFrame(data=polyDeg)
polySen = pd.DataFrame(data=polySen)
polyScore = pd.DataFrame(data=polyScore)
x_poly = pd.concat([polyDeg, polySen, polyScore], axis=1)
polyReg = LinearRegression()
polyReg.fit(x_poly, y)
PolyRegRes = polyReg.predict(x_poly)
print("Polynomial Regression - ", r2_score(y, PolyRegRes))
plt.title('Polynomial Regression')
plt.scatter(y, PolyRegRes, color='red')
plt.plot(y, y)
plt.show()

stdScale = StandardScaler()
xScaled = stdScale.fit_transform(x)
yScaled = stdScale.fit_transform(y)
svReg = SVR(kernel='rbf', gamma='auto')
svReg.fit(xScaled, yScaled.ravel())
svRegRes = svReg.predict(xScaled)
print("Support Vector Regression - ", r2_score(yScaled, svRegRes))
plt.title('Support Vector Regression')
plt.scatter(yScaled, svRegRes, color='red')
plt.plot(yScaled, yScaled)
plt.show()

treeReg = DecisionTreeRegressor()
treeReg.fit(x, y.ravel())
treeRegRes = treeReg.predict(x)
print("Decision Tree Regression - ", r2_score(y, treeRegRes))
plt.title('Decision Tree Regression')
plt.scatter(y, treeRegRes, color='red')
plt.plot(y, y)
plt.show()

forestReg = RandomForestRegressor(n_estimators=10)
forestReg.fit(x, y.ravel())
forestRegRes = forestReg.predict(x)
print("Random Forest Regression - ", r2_score(y, forestRegRes))
plt.title('Random Forest Regression')
plt.scatter(y, forestRegRes, color='red')
plt.plot(y, y)
plt.show()
