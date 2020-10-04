import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def runAll():
    df = pd.read_csv(os.path.join("outputs", "variance", "data.csv"))

    y = df['label']
    X = df.drop(columns=['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Sklearn accuracy:", accuracy_score(y_test, y_pred))
