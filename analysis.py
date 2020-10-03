from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import pandas as pd
from dtree import DecisionTreeClassifier, accuracy, test_train_split, confidence_interval
import random

OUTPUT_DIR = "outputs"
DATA = os.path.join(OUTPUT_DIR, "data.csv")


def makeTrees(X_train, y_train, X_test, y_test, start, stop):
    train_accuracies = OrderedDict()
    test_accuracies = OrderedDict()
    for i in range(start, stop):
        print("MaxHeight =", i)
        clf = DecisionTreeClassifier(max_height=i)
        X_grow, y_grow, X_val, y_val = test_train_split(X_train, y_train, 0.8, random_state=random.randint(100, 1000))
        clf.fit(X_grow, y_grow, attrs)

        #clf.plot(name="tree_" + str(i), storepath="outputs")

        clf.prune(X_val, y_val)

        #clf.plot(name="tree_pruned_" + str(i), storepath="outputs")

        y_pred = clf.predict(X_train)
        train_accuracies[i] = accuracy(y_train, y_pred)
        CI = confidence_interval(y_train, y_pred)
        print("Training Accuracy:", train_accuracies[i])
        print("Confidence interval on training sample:", CI)

        y_pred = clf.predict(X_test)
        test_accuracies[i] = accuracy(y_test, y_pred)
        CI = confidence_interval(y_test, y_pred)
        print("Test Accuracy:", test_accuracies[i])
        print("Confidence interval on test sample:", CI)
        #clf.save(name="tree_" + str(i), storepath="outputs")
        #clf.plot(name="tree_" + str(i), storepath="outputs")
    
    return train_accuracies, test_accuracies


if __name__ == "__main__":
    df = pd.read_csv(DATA)
    y = df["label"]
    X = df.drop(columns=["label"])
    print(df)
    attrs = [(a, "Continuous") for a in X.columns.tolist()]

    data_splits = []
    for _ in range(10):
        data_splits.append(test_train_split(X, y, 0.8, random_state=random.randint(100, 1000)))

    agg_acc = []

    i = 0
    for X_train, y_train, X_test, y_test in data_splits:
        print("Split", i, "\n\n")
        agg_acc.append(makeTrees(X_train, y_train, X_test, y_test, 2, 16))

    train_accs, test_accs = zip(*agg_acc)
    train_accuracy = OrderedDict()
    test_accuracy = OrderedDict()

    for i in train_accs[0].keys():
        for d in train_accs:
            if i not in train_accuracy:
                train_accuracy[i] = d[i] / (16 - 2)
            else:
                train_accuracy[i] += d[i] / (16 - 2)

        for d in test_accs:
            if i not in test_accuracy:
                test_accuracy[i] = d[i] / (16 - 2)
            else:
                test_accuracy[i] += d[i] / (16 - 2)

    plt.xlabel("Max Height")
    plt.ylabel("Accuracy")
    px1, py1 = zip(*list(train_accuracy.items()))
    plt.plot(px1, py1, label="Training Accuracy")
    px2, py2 = zip(*list(test_accuracy.items()))
    plt.plot(px2, py2, label="Test Accuracy")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
