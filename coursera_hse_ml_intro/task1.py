import pandas
from collections import defaultdict

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

sexes = data['Sex'].value_counts()
with open('1.txt', 'w+') as f:
    f.write("{males} {females}".format(males=sexes['male'],
                                       females=sexes['female']))

survived = data['Survived'].value_counts()
percent = survived[1] / data['Survived'].count()
print(percent)
with open('2.txt', 'w+') as f:
    f.write("{}".format(round(percent * 100, 2)))

classes = data['Pclass'].value_counts()
first = classes[1] / data['Pclass'].count() * 100
print(first)
with open('3.txt', 'w+') as f:
    f.write("{}".format(round(first, 2)))

ages = data['Age']
with open('4.txt', 'w+') as f:
    f.write("{} {}".format(round(ages.mean(), 2), ages.median()))

corr = data['SibSp'].corr(data['Parch'])
print(corr)
with open('5.txt', 'w+') as f:
    f.write("{}".format(round(corr, 2)))

names = list(data['Name'])
print(names)


def f(s):
    s = s.split()
    if 'Mrs.' in s:
        for i in range(s.index('Mrs.'), len(s)):
            if s[i].startswith("("):
                return s[i][1:].replace(')', '')
        return s[s.index('Mrs.') + 1]
    if 'Miss.' in s:
        return s[s.index('Miss.') + 1]


names = [x for x in names if 'Mrs.' in x or 'Miss.' in x]
names = [f(x) for x in names]
print(names)

res = defaultdict(int)
for e in names:
    res[e] += 1

with open('6.txt', 'w+') as f:
    f.write(sorted(res, key=lambda x: -res[x])[0])
print(sorted(res, key=lambda x: -res[x])[0])
print(res[sorted(res, key=lambda x: -res[x])[0]])
