import math

import numpy as np
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

data = pandas.read_csv('gbm-data.csv')
X = data.drop('Activity', axis=1)
y = data['Activity']
data = np.array(pandas.read_csv('gbm-data.csv').values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

for learning_rate in [0.2]:
    cls = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate)
    cls.fit(X_train, y_train)

    print(cls.learning_rate)
    sigma_func = lambda x: 1/(1+math.e**(-x))
    sdc_train = list(cls.staged_decision_function(X_train))
    sdc_test = list(cls.staged_decision_function(X_test))
    for i in range(250):
        pred_train = list(map(sigma_func, sdc_train[i]))
        pred_test = list(map(sigma_func, sdc_test[i]))
        loss_train = log_loss(y_train, pred_train)
        loss_test = log_loss(y_test, pred_test)
        print(i, loss_train, loss_test)


clf = RandomForestClassifier(n_estimators=36, random_state=241)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)

print(log_loss(y_test, pred))
