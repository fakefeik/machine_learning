import pandas
import numpy as np
# from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


csv = pandas.read_csv('task.csv')
# csv_filtered = csv[np.isfinite(csv['pidAggr'])]
csv_filtered = csv[(csv['pidAggr'] == 1) | (csv['pidAggr'] == 2)]
csv_unknown = csv[~np.isfinite(csv['pidAggr'])]
target = csv_filtered["pidAggr"]

data = csv_filtered.drop("pidAggr", axis=1).fillna(-1)
data_unknown = csv_unknown.drop("pidAggr", axis=1).fillna(-1)

cls = RandomForestClassifier(1000)
cls.fit(data.drop('id', axis=1), target)
a = cls.predict(data_unknown.drop('id', axis=1))

s = ''
for i in range(len(data_unknown)):
    s += '{},{}\n'.format(int(data_unknown.iloc[i]['id']), int(a[i]))
with open('RandomForest.txt', 'w+') as f:
    f.write(s)

c = cross_val_score(RandomForestClassifier(1000), data, target, cv=5)
print(c.mean(), c.std())
