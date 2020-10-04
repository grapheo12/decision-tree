import numpy as np
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

#randomly splits the input data in input ratio randomly based on input random state
#into train&test or train&validation, in pruning it is used for train & validation
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

#traverse the tree from root to leaf based on the input an dpredicts the output
def _predict(x, tree):
    #tmp = self.tree
    tmp = tree
    while not tmp._isLeaf:
        buff = tmp.descend(x)
        if buff is None:
            return tmp.majority
        tmp = buff
    return tmp.decision

#predicts for all input data
def predict(X, tree):
    y = X.apply(_predict, axis=1, tree=tree)
    return y

#a cut in the tree at given node
def cut(tree, node, type, X, y):
    #If type is temp, then cuts the tree at given node temporarily making it a leaf node
    #calculates the validation accuracy upon temporary removal of the node
    #restores back the node and returns the validation accuracy
    if type is "temp":
        initial_leaf = node._isLeaf         #store initial node type
        initial_decision = node._decision   #store initial decision
        node._isLeaf = True                 #changes node to leaf
        node._decision = node.majority      #sets node's decision to the majority
        y_pred = predict(X, tree)           #predicts on the temporarily pruned tree
        acc = accuracy(y,y_pred)            #calculates the validation accuracy

        node._isLeaf = initial_leaf         #restores initial node tpe
        node._decision = initial_decision   #restores initial decision
        return acc                          #returns accuracy
    #else, cuts the tree at given node permanently making it a leaf node
    #calculates and returns the validatoin accuracy upon cutting the node
    else:
        node._isLeaf = True                 #changes node to leaf type
        if node._decision is None:        
            node._decision = node.majority  #sets node's decision to majority
        del node.children                   #deletes all children of the node
        node.children = OrderedDict()       #creates an empty children list for the node
        y_pred = predict(X, tree)           #predicts on the permanently pruned tree
        acc = accuracy(y,y_pred)            #calculates the validation accuracy
        return acc                          #returns accuracy

#naive pruning algorithm
#tests every non-leaf node for increase in validation accuracy
#prunes the one with greatest increase
def prune(tree, X, y, training_ratio, random_state):
    
    X_val, y_val = X, y
    node = tree
    while(1):
        #calculate and store the initial validation accuracy before pruning
        y_pred = predict(X_val, tree)
        initial_val_accuracy = accuracy(y_val, y_pred)
        print("Initial validation accuracy:",initial_val_accuracy)

        #initialize max_prune_accuracy to 0 and max__prune_node to None
        max_prune_accuracy = 0.0
        max_prune_node = None
        
        queue = []          #queue for performing bfs of the tree
        visited = set()     #keep track of visited nodes during bfs
        queue.append(tree)  #qppend the root node
        
        while(queue):                   #loop till queue becomes empty
            node  = queue.pop(0)        #pop queue's front node
            visited.add(node)           #mark it visited
            if node._isLeaf:            #if it is a leaf node then continue
                continue
            else:                       #else check if validation accuracy increases on pruning, if yes then prune
                acc = cut(tree,node,'temp', X_val, y_val)       #store validation accuracy on temporary pruning of the node
                if acc>max_prune_accuracy:                      #if acc>max_prune_accuracy, update max_prune_accuracy and max_prune_node
                    max_prune_accuracy = acc
                    max_prune_node = node
                for value in node.children:                     #push all children of that node in the queue
                    child = node.children[value]
                    if child not in visited and not child._isLeaf:
                        queue.append(child)
        print("max prune accuracy:",max_prune_accuracy)
        if max_prune_accuracy <= initial_val_accuracy:          #if max_prune_accuracy among all nodes <= initial_val_accuracy, stop pruning
            print("cant prune")
            break
        else:
            if max_prune_node is not None:                      #else permanently prune at max_prune_node
                print("cutting")
                cut(tree,max_prune_node,'perm', X_val, y_val)
    return tree     #return the pruned tree

#bottom-up pruning algorithm
#starts pruning from leaf nodes' parent
#prunes till validation accuracy increases
#or no pruning possible at any of leaf nodes' parent
def prune2(tree, node, X_val, y_val):
	children = node.children   #children of current node
	no_leaves = 0              #no of leaf-children of current node
	for key, child in children.items():#for every child node
		if not child._isLeaf:          #if it is non-leaf node call prune at that node
			child = prune2(tree, child, X_val, y_val)
		if child._isLeaf:              #if it is  leaf node increase leaf-count
			no_leaves += 1
	if no_leaves==0:   #if the node has no children left after pruning the subtrees rooted at that node
		return node    #then stop pruning and return that node
	else:
		y_pred = predict(X_val, tree)                     
		initial_val_accuracy = accuracy(y_val, y_pred)#store initial validation accuracy before pruning at current node
		initial_leaf = node._isLeaf                   
		initial_decision = node._decision
		node._isLeaf = True              #temporarily prune the current node
		node._decision = node.majority   #temporarily assign node's decision to majority at that node
		y_pred = predict(X_val, tree)
		acc = accuracy(y_val, y_pred)    #calculated validation accuracy on the temporarily pruned node

		if acc>initial_val_accuracy:     #if validation accuracy increases on pruning
			del node.children            #permanently prune the node
			node.children = OrderedDict()
			return node                  #return current node
		else:                            #else restore initial node values
			node._isLeaf = initial_leaf
			node._decision = initial_decision
			return node                  #return current node
