import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
df = pd.read_csv('Restaurant_Reviews.csv')
print(df)

comment0 = re.sub('[^a-zA-Z]', ' ', df['Review'][0])
print(comment0)
comment0 = comment0.lower()
print(comment0)
comment0 = comment0.split()
print(comment0)

ps = PorterStemmer()
comment0 = [ps.stem(word) for word in comment0 if word not in set(stopwords.words('english'))]
print(comment0)
comment0 = ' '.join(comment0)
print(comment0)

clearReviews = []
for i in range(1000):
    comment = re.sub('[^a-zA-Z]', ' ', df['Review'][i]).lower().split()
    comment = [ps.stem(word) for word in comment if word not in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    clearReviews.append(comment)

print(clearReviews)
cv = CountVectorizer(max_features=2000)
clearReviews = cv.fit_transform(clearReviews).toarray()
print(clearReviews)
likes = df.iloc[:, [1]].values

x_train, x_test, y_train, y_test = train_test_split(clearReviews, likes, test_size=.33)
gnb = GaussianNB()
gnb.fit(x_train, y_train.ravel())
res = gnb.predict(x_test)

cfm = confusion_matrix(y_test, res)
print(cfm)
