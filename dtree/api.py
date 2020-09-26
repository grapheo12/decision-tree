import math
import os
import pickle
from graphviz import Digraph
import numpy as np
from dtree.learner import id3Categorical

MAX_HEIGHT = 100
MAX_NODES = 1000000


class DecisionTreeClassifier:
    def __init__(self, max_height=MAX_HEIGHT):
        if max_height == -1:
            self.max_height = MAX_HEIGHT
        else:
            self.max_height = max_height

        self.tree = None
        self.attrs = None

    def fit(self, X, y, attrs):
        self.attrs = attrs

        self.tree = id3Categorical(X, y, attrs, y.index.tolist(),
                                   iter(range(MAX_NODES)),
                                   0, self.max_height)

    def _predict(self, x):
        tmp = self.tree
        while not tmp.isLeaf:
            buff = tmp.descend(x)
            if buff is None:
                return tmp.majority
            tmp = buff
        return tmp.decision

    def predict(self, X):
        y = X.apply(self._predict, axis=1)
        return y

    def plot(self, name="Test_Tree", storepath="tests", view=True):
        dot = Digraph(comment=name)
        dot.attr(rankdir="LR")

        for parent, relation, node in self.tree.preorderTraverse():
            if node.isLeaf:
                dot.node(str(node.idx), "Decision: " + str(node.decision))
            else:
                dot.node(str(node.idx), str(node.discriminator_text))

            if parent is not None:
                dot.edge(str(parent.idx), str(node.idx), label=str(relation))

        fname = os.path.join(storepath, name + ".gv")

        dot.render(fname, view=view)

    def save(self, name="Test_Tree", storepath="tests"):
        with open(os.path.join(storepath, name + ".pickle"), "wb") as f:
            pickle.dump(self, f)


def accuracy(y_true, y_pred):
    values = (y_true == y_pred).value_counts()
    try:
        return values[True] / (values[True] + values[False])
    except Exception:
        print(values)
        if False not in values:
            return 1.0
        else:
            return 0


def test_train_split(X, y, training_ratio, random_state=0):
    idx = X.index.to_numpy()
    np.random.seed(random_state)

    np.random.shuffle(idx)
    n = idx.shape[0]
    X_train = X.loc[idx[:math.floor(n * training_ratio)]]
    X_test = X.loc[idx[math.floor(n * training_ratio):]]
    y_train = y.loc[idx[:math.floor(n * training_ratio)]]
    y_test = y.loc[idx[math.floor(n * training_ratio):]]

    return X_train, y_train, X_test, y_test
