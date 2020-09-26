from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import pandas as pd
from dtree import DecisionTreeClassifier, accuracy, test_train_split

OUTPUT_DIR = "outputs"
DATA = os.path.join(OUTPUT_DIR, "data.csv")


def makeTrees(X_train, y_train, X_test, y_test, start, stop):
    train_accuracies = OrderedDict()
    test_accuracies = OrderedDict()
    for i in range(start, stop):
        print("MaxHeight =", i)
        clf = DecisionTreeClassifier(max_height=i)
        clf.fit(X_train, y_train, attrs)

        y_pred = clf.predict(X_train)
        train_accuracies[i] = accuracy(y_train, y_pred)
        print("Training Accuracy:", train_accuracies[i])

        y_pred = clf.predict(X_test)
        test_accuracies[i] = accuracy(y_test, y_pred)
        print("Test Accuracy:", test_accuracies[i])

        # clf.save(name="tree_" + str(i), storepath="outputs")
        clf.plot(name="tree_" + str(i), storepath="outputs")

    plt.xlabel("Max Height")
    plt.ylabel("Accuracy")
    px1, py1 = zip(*list(train_accuracies.items()))
    plt.plot(px1, py1, label="Training Accuracy")
    px2, py2 = zip(*list(test_accuracies.items()))
    plt.plot(px2, py2, label="Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    y = df["label"]
    X = df.drop(columns=["label"])
    attrs = [(a, "Continuous") for a in X.columns.tolist()]

    X_train, y_train, X_test, y_test = test_train_split(X, y, 0.8, random_state=42)
    makeTrees(X_train, y_train, X_test, y_test, 2, 11)
