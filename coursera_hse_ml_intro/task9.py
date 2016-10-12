import pandas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)

c = pandas.read_csv('classification.csv')
r = lambda d: round(d, 2)

print(
    len(c[(c['true'] == 1) & (c['pred'] == 1)]),
    len(c[(c['true'] == 0) & (c['pred'] == 1)]),
    len(c[(c['true'] == 1) & (c['pred'] == 0)]),
    len(c[(c['true'] == 0) & (c['pred'] == 0)])
)

print(
    r(accuracy_score(c['true'], c['pred'])),
    r(precision_score(c['true'], c['pred'])),
    r(recall_score(c['true'], c['pred'])),
    r(f1_score(c['true'], c['pred']))
)

s = pandas.read_csv('scores.csv')
cls = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']
for e in cls:
    print(r(roc_auc_score(s['true'], s[e])), end='')
print()

max = ('', 0)
for e in cls:
    c = precision_recall_curve(s['true'], s[e])
    precision = c[0]
    recall = c[1]
    thresholds = c[2]
    for i in range(len(precision)):
        if recall[i] >= 0.7:
            if precision[i] > max[1]:
                max = e, precision[i]
print(max[0])
