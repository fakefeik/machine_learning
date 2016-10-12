import pandas
from sklearn.svm import SVC

csv = pandas.read_csv('svm-data.csv', header=None, names=["T", "P1", "P2"])
target = csv["T"]
data = csv.drop("T", axis=1)
cls = SVC(C=100000, kernel='linear', random_state=241)
cls.fit(data, target)
print(cls.support_)
