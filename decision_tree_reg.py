import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('salaries.csv')
print(df)

level = df.iloc[:, [1]]
salary = df.iloc[:, [2]]
dtReg = DecisionTreeRegressor(random_state=0)
dtReg.fit(level, salary)

plt.scatter(level, salary)
plt.plot(level, dtReg.predict(level))
plt.show()
