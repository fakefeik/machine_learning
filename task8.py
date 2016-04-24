import pandas
from math import sqrt, e
from sklearn.metrics import roc_auc_score


logistics = pandas.read_csv('data-logistic.csv', header=None, names=["T", "P1", "P2"])
r = lambda d: round(d, 3)


def gradient_descent(X, y, C=10, k=0.1, w=(0, 0)):
    l = len(X)
    return (
        w[0] + k * 1/l * sum(map(lambda i: y[i] * X['P1'][i]*(1-1/(1+pow(e, -y[i] * (w[0] * X['P1'][i] + w[1] * X['P2'][i])))), range(l))) - k * C * w[0],
        w[1] + k * 1/l * sum(map(lambda i: y[i] * X['P2'][i]*(1-1/(1+pow(e, -y[i] * (w[0] * X['P1'][i] + w[1] * X['P2'][i])))), range(l))) - k * C * w[1],
    )

X = logistics.drop('T', axis=1)
y = logistics['T']

w = (0, 0)
print(gradient_descent(X, y, 0))
for i in range(10000):
    w_old = w
    w = gradient_descent(X, y, w=w)
    if sqrt(pow(w_old[0]-w[0], 2)+pow(w_old[1]-w[1], 2)) <= 1e-5:
        break
print(w)

lines = ['T,P\n']
for i in range(len(logistics['T'])):
    lines.append('{}, {}\n'.format(logistics['T'][i], 1/(1+pow(e, -w[0]*X['P1'][i] - w[1]*X['P2'][i]))))

with open('res.txt', 'w+') as f:
    f.writelines(lines)

res = pandas.read_csv('res.txt')

print(r(roc_auc_score(res['T'], res['P'])))
