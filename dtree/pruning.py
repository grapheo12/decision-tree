import numpy as np
import pandas as pd
import copy
from dtree.tree import DecisionTreeNode
from collections import OrderedDict

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

def _predict(x, tree):
    #tmp = self.tree
    tmp = tree
    while not tmp._isLeaf:
        buff = tmp.descend(x)
        if buff is None:
            return tmp.majority
        tmp = buff
    return tmp.decision

def predict(X, tree):
    y = X.apply(_predict, axis=1, tree=tree)
    return y

def cut(tree, node, type, X, y):
    if type is "temp":
        initial_leaf = node._isLeaf
        initial_decision = node._decision
        node._isLeaf = True
        node._decision = node.majority
        y_pred = predict(X, tree)
        acc = accuracy(y,y_pred)

        node._isLeaf = initial_leaf
        node._decision = initial_decision
        return acc
    else:
        node._isLeaf = True
        if node._decision is None:
            node._decision = node.majority
        del node.children
        node.children = OrderedDict()
        y_pred = predict(X, tree)
        acc = accuracy(y,y_pred)
        return acc

def prune(tree, X, y, training_ratio, random_state):
    
    #X_train, y_train, X_val, y_val = test_train_split(X, y, training_ratio, random_state=random.randint(100, 1000))
    X_val, y_val = X, y
    node = tree
    #return node
    prev_val_accuracy = 0.0
    while(1):
        y_pred = predict(X_val, tree)
        initial_val_accuracy = accuracy(y_val, y_pred)
        print("Initial validation accuracy:",initial_val_accuracy)

        max_prune_accuracy = 0.0
        max_prune_node = None
        #cut and test here
        queue = []
        visited = set()
        queue.append(tree)
        #visited.add(tree)
        while(queue):
            node  = queue.pop(0)
            visited.add(node)
            if node._isLeaf:
                continue
            else:
                acc = cut(tree,node,'temp', X_val, y_val)
                #print("acc:",acc)
                if acc>max_prune_accuracy:
                    max_prune_accuracy = acc
                    max_prune_node = node
                for value in node.children:
                    child = node.children[value]
                    if child not in visited and not child._isLeaf:
                        queue.append(child)
        print("max prune accuracy:",max_prune_accuracy)
        if max_prune_accuracy <= initial_val_accuracy:
            print("cant prune")
            break
        else:
            if max_prune_node is not None:
                print("cutting")
                cut(tree,max_prune_node,'perm', X_val, y_val)
    return tree

def prune2(tree, node, X_val, y_val):
	children = node.children
	no_leaves = 0
	for key, child in children.items():
		if not child._isLeaf:
			child = prune2(tree, child, X_val, y_val)
		if child._isLeaf:
			no_leaves += 1
	if no_leaves==0:
		return node
	else:
		y_pred = predict(X_val, tree)
		initial_val_accuracy = accuracy(y_val, y_pred)
		initial_leaf = node._isLeaf
		initial_decision = node._decision
		node._isLeaf = True
		node._decision = node.majority
		y_pred = predict(X_val, tree)
		acc = accuracy(y_val, y_pred)

		if acc>initial_val_accuracy:
			del node.children
			node.children = OrderedDict()
			return node
		else:
			node._isLeaf = initial_leaf
			node._decision = initial_decision
			return node