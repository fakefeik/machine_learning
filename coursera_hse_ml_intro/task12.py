import pandas
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

data = pandas.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.drop('Rings', axis=1)
y = data['Rings']

for i in range(1, 50):
    r = RandomForestRegressor(n_estimators=i, random_state=1)
    ng = KFold(len(y), n_folds=5, shuffle=True, random_state=1)
    s = cross_val_score(r, X, y, cv=ng)
    print(i, s.mean())
