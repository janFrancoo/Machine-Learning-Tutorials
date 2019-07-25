import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

df = pd.read_csv('Ads_CTR_Optimisation.csv')
print(df)

rows, columns = df.shape
totalReward = 0
selected = []
for n in range(rows):
    ad = np.random.randint(0, columns)
    selected.append(ad)
    reward = df.values[n, ad]
    totalReward += reward

plt.title('In Random Total Reward = ' + str(totalReward))
plt.hist(selected)
plt.show()

rewards = [0] * columns
clicked = [0] * columns
totalReward = 0
selected = []
for n in range(rows):
    ad = 0
    max_ucb = 0
    for i in range(columns):
        if clicked[i] > 0:
            avg = rewards[i] / clicked[i]
            delta = math.sqrt(3/2 * math.log(n) / clicked[i])
            ucb = avg + delta
        else:
            ucb = rows * 10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    selected.append(ad)
    clicked[ad] = clicked[ad] + 1
    reward = df.values[n, ad]
    rewards[ad] += reward
    totalReward += reward

plt.title('UCB Total Reward = ' + str(totalReward))
plt.hist(selected)
plt.show()
