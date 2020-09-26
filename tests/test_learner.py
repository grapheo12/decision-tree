import pandas as pd
from dtree.learner import entropy, id3Categorical
from dtree.api import plot


def test_entropy():
    df = pd.read_csv("tests/test.csv")
    print(df)
    print(entropy(df["Pclass"], list(range(1, 100))))


def test_id3Categorical():
    df = pd.read_csv("tests/test.csv")
    y = df["Pclass"]
    X = df.drop(columns=["Pclass", "Name", "PassengerId",
                         "Ticket", "Cabin", "Fare"])
    attrs = [(a, "categorical") if a != "Age" else (a, "Continuous")
             for a in X.columns.tolist()]
    root = id3Categorical(X, y, attrs, y.index.tolist(),
                          iter(range(10000)), 0, 6)
    max_height = -1
    for parent, relation, node in root.preorderTraverse():
        max_height = max([max_height, node.height])
        if parent is not None:
            print(parent.idx, relation, node.idx, node.height,
                  node.discriminator_text, node.decision)
        else:
            print(None, relation, node.idx, node.height,
                  node.discriminator_text, node.decision)

    print(max_height)
    plot(root)
