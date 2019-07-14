import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('salaries.csv')
print(df)

level = df.iloc[:, [1]].values
salary = df.iloc[:, [2]].values

rfReg = RandomForestRegressor(n_estimators=10, random_state=0)
rfReg.fit(level, salary)

plt.scatter(level, salary)
plt.plot(level, rfReg.predict(level))
plt.show()
