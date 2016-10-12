import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv("titanic.csv",  index_col='PassengerId')
df = pandas.DataFrame(data, columns=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'])
df_filtered = df.dropna()
df_filtered['Sex'] = df_filtered['Sex'].map(lambda x: 1 if x == 'male' else 0)

target = df_filtered['Survived']
df_filtered = df_filtered.drop('Survived', axis=1)

print(df_filtered)

clf = DecisionTreeClassifier(random_state=241)
print(clf.fit(df_filtered, target))

print(clf.feature_importances_)
