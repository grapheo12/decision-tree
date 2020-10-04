from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import pandas as pd
from dtree import DecisionTreeClassifier, accuracy, test_train_split, confidence_interval
import random

OUTPUT_DIR = os.path.join("outputs", "rolling_mean")
DATA = os.path.join(OUTPUT_DIR, "data.csv")
NUM_SPLITS = 10


def makeMaxHeightTree(X_train, y_train, X_test, y_test, attrs):
    clf = DecisionTreeClassifier(max_height=-1)
    clf.fit(X_train, y_train, attrs)
    y_pred = clf.predict(X_train)
    train_accuracy = accuracy(y_train, y_pred)
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy(y_test, y_pred)

    return train_accuracy, test_accuracy, clf


def makeTrees(X_train, y_train, X_test, y_test, attrs, start, stop):
    train_accuracies = OrderedDict()
    test_accuracies = OrderedDict()
    clfs = OrderedDict()
    for i in range(start, stop):
        print("MaxHeight =", i)
        clf = DecisionTreeClassifier(max_height=i)
        clf.fit(X_train, y_train, attrs)

        clfs[i] = clf

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
    return train_accuracies, test_accuracies, clfs


def runAll():
    print("Reading data and splitting into training, validation and test sets.")
    df = pd.read_csv(DATA)
    y = df["label"]
    X = df.drop(columns=["label"])
    attrs = [(a, "Continuous") for a in X.columns.tolist()]

    data_splits = []
    for _ in range(NUM_SPLITS):
        data_splits.append(test_train_split(X, y, 0.8, random_state=random.randint(100, 1000)))

    print("Part 1: Tree with full height")

    results = []
    i = 0
    for X_train, y_train, X_test, y_test in data_splits:
        print("Generating for training set:", i)
        i += 1

        results.append(makeMaxHeightTree(X_train, y_train, X_test, y_test, attrs))

    train_accs, test_accs, clfs = zip(*results)
    print("Average training accuracy:", sum(train_accs) / len(train_accs))
    print("Max training accuracy:", max(train_accs))
    print("Average test accuracy:", sum(test_accs) / len(test_accs))
    print("Max test accuracy:", max(test_accs))

    print("Saving tree with best test accuracy")
    maxIdx = test_accs.index(max(test_accs))
    clfs[maxIdx].plot(name="Max_height", storepath=OUTPUT_DIR)
    print("Part 1 End")

    print("Part 2: Best Depth Selection")

    X_train, y_train, X_test, y_test = data_splits[0]
    X_grow, y_grow, X_val, y_val = test_train_split(X_train, y_train, 0.8, random_state=random.randint(100, 1000))
    train_accuracy, test_accuracy, clfs = makeTrees(X_grow, y_grow, X_test, y_test, attrs, 2, 16)

    plt.xlabel("Max Height")
    plt.ylabel("Accuracy")
    px1, py1 = zip(*list(train_accuracy.items()))
    plt.plot(px1, py1, label="Training Accuracy")
    px2, py2 = zip(*list(test_accuracy.items()))
    plt.plot(px2, py2, label="Test Accuracy")
    plt.legend()

    print("Saving accuracy vs height plot")
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
    print("Part 2 End")

    BEST_HEIGHT = 10  # Obtained from the accuracy graph

    print("Part 3: Pruning the best height tree")
    clf = clfs[BEST_HEIGHT]
    clf.prune(X_val, y_val)

    y_pred = clf.predict(X_test)
    print("Accuracy after pruning:", accuracy(y_test, y_pred))
    print("Print 3 End")

    print("Part 4: Printing the Decision Tree")
    print("Saving the tree as Final_tree.pdf")
    clf.plot(name="Final_tree", storepath=OUTPUT_DIR)
    print("Part 4 End")
