import apyori
import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

df = pd.read_csv('sepet.csv', header=None)
print(df)

t = []
for i in range(7501):
    t.append([str(df.values[i, j]) for j in range(20)])

rules = apyori.apriori(t, min_support=0.01, min_confidence=0.2, min_lift=3, min_length=2)
print(list(rules))
