import pandas
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()

train = pandas.read_csv('perceptron-train.csv', header=None, names=["T", "P1", "P2"])
train_target = train["T"]
train_data = train.drop("T", axis=1)

test = pandas.read_csv('perceptron-test.csv', header=None, names=["T", "P1", "P2"])
test_target = test["T"]
test_data = test.drop("T", axis=1)

p = Perceptron(random_state=241)
p.fit(train_data, train_target)
predictions = p.predict(test_data)
a = accuracy_score(test_target, predictions)
print(a)

train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

p.fit(train_data, train_target)
predictions = p.predict(test_data)
b = accuracy_score(test_target, predictions)
print(b)
print(b - a)
