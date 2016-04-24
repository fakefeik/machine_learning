import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score


wine = pandas.read_csv('wine.data')
target = wine['Class']
count = wine['Class'].count()
wine = wine.drop('Class', axis=1)

wine = scale(wine)
for k in range(1, 51):
    gen = KFold(count, n_folds=5, shuffle=True, random_state=42)
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), wine, target, cv=gen)
    print(k)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
