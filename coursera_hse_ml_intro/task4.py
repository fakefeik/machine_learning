from numpy import linspace
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score

boston = load_boston()
boston.data = scale(boston.data)
g = {}
for i in linspace(1, 10, 200):
    gen = KFold(len(boston.data), n_folds=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        KNeighborsRegressor(weights='distance', p=i),
        boston.data, boston.target,
        scoring='mean_squared_error',
        cv=gen
    )
    g[i] = scores.mean()

k = sorted(g, key=lambda x: -g[x])[0]
v = g[k]
print(k, v)
