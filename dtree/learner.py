from functools import partial
import math
from multiprocessing import Pool
import numpy as np

from dtree.tree import DecisionTreeNode

MAX_POOL_SIZE = 10
ENTROPY_PRECISION = 0.1
MIN_BIN_SIZE = 10
MAX_ITER = 2


def fastMap(mapper, data):
    i = 0
    ans = []
    while i < len(data):
        with Pool(MAX_POOL_SIZE) as pool:
            ans.extend(pool.map(mapper, data[i:i+MAX_POOL_SIZE]))
        i += MAX_POOL_SIZE

    return ans


def entropy(data, idxList):
    df = data.loc[idxList]
    counts = df.value_counts().to_numpy()
    counts = counts.reshape(1, -1).astype(np.float32)
    counts /= np.sum(counts)
    log_sum = counts @ np.log2(counts.T)
    return -log_sum[0, 0]


def entropyWithBestSplit(attr, X, y):
    df = X[attr]
    maxVal = df.max()
    minVal = df.min()

    # Square root decomposition
    r = math.ceil(math.sqrt(maxVal - minVal))
    entropyVal = float('inf')
    splitAt = -1

    iterNum = 0
    while r > MIN_BIN_SIZE and iterNum < MAX_ITER:
        iterNum += 1
        i = minVal
        while i < maxVal:
            leftSplit = df[df < i].index.tolist()
            rightSplit = df[df >= i].index.tolist()
            buff_entropy = entropy(df, leftSplit) *\
                (len(leftSplit) / (len(leftSplit) + len(rightSplit)))
            buff_entropy += entropy(df, rightSplit) *\
                (len(rightSplit) / (len(leftSplit) + len(rightSplit)))
            if entropyVal >= buff_entropy:
                entropyVal = buff_entropy
                splitAt = i
            i += r
        minVal = min([splitAt - r, minVal])
        maxVal = max([splitAt + r, maxVal])
        r_buff = math.ceil(math.sqrt(maxVal - minVal))
        if r_buff >= r:
            break
        else:
            r = r_buff
    finalIdxLists = [
        df[df < splitAt].index.tolist(),
        df[df >= splitAt].index.tolist()
    ]
    return [splitAt], finalIdxLists, entropyVal


def entropyCategorical(attr, X, y):
    uniques = X[attr].unique().tolist()
    idxLists = []
    entropies = []
    weights = []
    for u in uniques:
        idxLists.append(X.index[X[attr] == u].tolist())
        entropies.append(entropy(y, idxLists[-1]))
        weights.append(len(idxLists[-1]))

    entropies = np.array(entropies).reshape(1, -1)
    weights = np.array(weights).reshape(-1, 1).astype(np.float32)
    weights /= np.sum(weights)

    return (uniques, idxLists, (entropies @ weights)[0, 0])


def listEntropyFactory(attr, X, y):
    attribute, mode = attr
    if mode == "categorical":
        return entropyCategorical(attribute, X, y)
    else:
        return entropyWithBestSplit(attribute, X, y)


def id3Categorical(X, y, attrs, idxList, idGen, height, maxHeight):
    unique_value_counts = y.loc[idxList].nunique()
    print(unique_value_counts, end="\r")
    if unique_value_counts == 1:
        # Homogenous node
        node = DecisionTreeNode(idx=next(idGen), discriminator=None,
                                discriminator_text=None,
                                height=height)
        node.decision = y.loc[idxList[0]]
        return node
    elif unique_value_counts == 0:
        return None

    entropyLists = fastMap(partial(listEntropyFactory,
                                   X=X.loc[idxList], y=y.loc[idxList]),
                           attrs)

    uniques, idxLists, entropies = zip(*entropyLists)

    curr_entropy = entropy(y, idxList)

    info_gains = np.array(entropies) - curr_entropy
    info_gains = -info_gains

    best_attr = np.argmax(info_gains)

    unique_values = uniques[best_attr]
    req_idxLists = idxLists[best_attr]

    if attrs[best_attr][1] == "categorical":
        node = DecisionTreeNode(idx=next(idGen), discriminator=lambda x: x[attrs[best_attr][0]],
                                discriminator_text=attrs[best_attr][0],
                                height=height)
    else:
        node = DecisionTreeNode(idx=next(idGen),
                                discriminator=lambda x: x[attrs[best_attr][0]] >= unique_values[0],
                                discriminator_text=
                                str(attrs[best_attr][0]) +
                                " >= " + str(unique_values[0]),
                                height=height)

    if (
        height < maxHeight and
        not np.isnan(info_gains).any() and
        (np.abs(info_gains) > ENTROPY_PRECISION).any()
    ):
        if attrs[best_attr][1] == "categorical":
            for i in range(len(unique_values)):
                node.addChild(key=unique_values[i],
                              node=id3Categorical(X, y, attrs, req_idxLists[i],
                                                  idGen, height + 1,
                                                  maxHeight))
        else:
            chnode1 = id3Categorical(X, y, attrs, req_idxLists[0],
                                     idGen, height + 1, maxHeight)
            chnode2 = id3Categorical(X, y, attrs, req_idxLists[1],
                                     idGen, height + 1, maxHeight)
            if chnode1 is not None and chnode2 is not None:
                node.addChild(key=False, node=chnode1)
                node.addChild(key=True, node=chnode2)
            else:
                node.decision = y.loc[idxList].value_counts().idxmax()

        node.majority = y.loc[idxList].value_counts().idxmax()

    else:
        node.decision = y.loc[idxList].\
                        value_counts().idxmax()

    return node
