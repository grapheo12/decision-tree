import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("outputs/data_var.csv")

y = df['label']
X = df.drop(columns=['label'])


X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
