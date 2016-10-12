import time

import pandas
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, cross_val_score


# Подход 1: градиентный бустинг "в лоб"

features = pandas.read_csv("features.csv", index_col='match_id')
y_train = features['radiant_win']
to_delete = ['radiant_win', 'duration', 'tower_status_radiant',
             'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
features.drop(to_delete, inplace=True, axis=1)

length = len(features)
data_with_skip = features.count()[features.count() != length]
print(data_with_skip)

X_train = features.fillna(0)

gen = KFold(length, n_folds=5, shuffle=True)
clf = GradientBoostingClassifier(n_estimators=30)
clf.fit(X_train, y_train)
start_time = time.time()
print(cross_val_score(clf, X_train, y_train, cv=gen, scoring='roc_auc').mean())
print(time.time() - start_time)


# Подход 2: логистическая регрессия

# scale features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
gen = KFold(length, n_folds=5, shuffle=True)
c = 0.01
clf = LogisticRegression(C=c)
clf.fit(X_train, y_train)
print(cross_val_score(clf, X_train, y_train, cv=gen, scoring='roc_auc').mean())

# delete cateorial features
categorial = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
                  'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
features.drop(categorial, inplace=True, axis=1)
X_train = features.fillna(0)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
clf.fit(X_train, y_train)
print(cross_val_score(clf, X_train, y_train, cv=gen, scoring='roc_auc').mean())

# unique ids
features = pandas.read_csv("features.csv", index_col='match_id')
to_delete = ['radiant_win', 'duration', 'tower_status_radiant',
             'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
features.drop(to_delete, inplace=True, axis=1)
heroes = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
          'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
ids = set()
for field in heroes:
    l = features[field].unique().tolist()
    for id in l:
        ids.add(id)
unique_ids = len(ids)
print("Unique ids " + str(unique_ids))

# bag of words
N = 113
length = len(features)
X_pick = np.zeros((length, N))
features = pandas.read_csv("features.csv", index_col='match_id')
to_delete = ['radiant_win', 'duration', 'tower_status_radiant',
             'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
features.drop(to_delete, inplace=True, axis=1)
for i, match_id in enumerate(features.index):
    for p in range(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
X_new = np.concatenate([X_train, X_pick], axis=1)
gen = KFold(length, n_folds=5, shuffle=True)
clf = LogisticRegression(penalty='l2', C=c)
clf.fit(X_new, y_train)
print(cross_val_score(clf, X_new, y_train, cv=gen, scoring='roc_auc').mean())

# predict
test_data = pandas.read_csv("features_test.csv", index_col='match_id')
features = pandas.read_csv("features_test.csv", index_col='match_id')
length = len(features)
test_data.drop(categorial, inplace=True, axis=1)
filled_data = test_data.fillna(0)
X_test = filled_data
scaler.fit(X_test)
X_test = scaler.transform(X_test)
X_pick = np.zeros((length, N))
for i, match_id in enumerate(features.index):
    for p in range(5):
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
X_new = np.concatenate([X_test, X_pick], axis=1)

answer = clf.predict_proba(X_new)[:, 1]
id_match = features.index
print('match_id, radiant_win')
min = 1
max = 0
for i in range(0, length):
    current = answer[i]
    print(str(id_match[i]) + ", " + str(current))
    if current > max:
        max = current
    if current < min:
        min = current
print('Min: ' + str(min))
print('Max: ' + str(max))
