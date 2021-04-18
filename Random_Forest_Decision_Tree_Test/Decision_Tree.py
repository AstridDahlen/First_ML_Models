import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv('movies1.csv')

labels = df1['score']
data = df1[['budget', 'gross', 'runtime','votes','year']]






# kolla så att de inte fattas data med null och na om det finns null/na värden droppa
# nans = df1[df1.isna().any(axis=1)]
# print(nans)
# df = df.dropna()
# df.isna().sum()



train_data, test_data, train_labels, test_labels = train_test_split(data, labels,random_state=1)


scores = []
for i in range(1,23):
    tree = DecisionTreeRegressor(random_state=1, max_depth=i)
    tree.fit(train_data,train_labels)
    score = tree.score(test_data, test_labels)
    scores.append(score)
plt.plot(range(1,23), scores)
print(scores)
plt.show()