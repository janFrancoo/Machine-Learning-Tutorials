import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('sales.csv')
months = df[['Aylar']]
sales = df[['Satislar']]

x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size=.33, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)
results = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, results)
plt.title("Sales by Months")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()
