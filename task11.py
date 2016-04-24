import pandas
import numpy as np
from sklearn.decomposition import PCA

data = pandas.read_csv('close_prices.csv').drop('date', axis=1)
djia = pandas.read_csv('djia_index.csv').drop('date', axis=1)

pca = PCA(n_components=10)
pca.fit(data)
tr = pca.transform(data)

s = 0
for i, e in enumerate(sorted(pca.explained_variance_ratio_, key=lambda x: -x)):
    s += e
    if s >= 0.9:
        print(i + 1)
        break

print(np.corrcoef(list(map(lambda x: x[0], tr)), djia['^DJI']))
print(list(data.columns.values)[list(pca.components_[0]).index(sorted(pca.components_[0], key=lambda x: -x)[0])])
