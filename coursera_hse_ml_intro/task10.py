import pandas
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

r = lambda d: str(round(d, 2))

train = pandas.read_csv('salary-train.csv')
train['FullDescription'] = train['FullDescription'].map(str.lower).replace('[^a-zA-Z0-9]', ' ', regex=True)
train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

test = pandas.read_csv('salary-test-mini.csv')
test['FullDescription'] = test['FullDescription'].map(str.lower).replace('[^a-zA-Z0-9]', ' ', regex=True)
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)

X_train_target = train['SalaryNormalized']

v = TfidfVectorizer(min_df=5)
X_train_tfid = v.fit_transform(train['FullDescription'])
X_test_tfid = v.transform(test['FullDescription'])

enc = DictVectorizer()
X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

ridge = Ridge(random_state=241)
ridge.fit(hstack([X_train_tfid, X_train_categ]), X_train_target)
pr = ridge.predict(hstack([X_test_tfid, X_test_categ]))
print(' '.join(map(r, pr)))
