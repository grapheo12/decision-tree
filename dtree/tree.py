from collections import OrderedDict


class DecisionTreeNode:
    def __init__(self, idx, discriminator,
                 discriminator_text="Node", height=0):

        self.idx = idx
        self.discriminator = discriminator
        self.discriminator_text = discriminator_text
        self.height = height
        self.children = OrderedDict()
        self._isLeaf = False
        self._decision = None
        self.majority = None

    def addChild(self, key, node, replace=False):
        if not replace and key in self.children:
            raise Exception("Child with same key exists")
        if node is not None:
            self.children[key] = node

    def descend(self, arg):
        value = self.discriminator(arg)
        if value in self.children:
            return self.children[value]
        else:
            return None

    def preorderTraverse(self, parent=None, parentRelation=None):
        yield (parent, parentRelation, self)
        for key, child in self.children.items():
            yield from child.preorderTraverse(parent=self, parentRelation=key)

    @property
    def isLeaf(self):
        return self._isLeaf

    @isLeaf.setter
    def isLeaf(self, val):
        raise NotImplementedError

    @property
    def decision(self):
        return self._decision

    @decision.setter
    def decision(self, label):
        if not self._isLeaf and self._decision is None:
            self._decision = label
            self._isLeaf = True
        else:
            raise NotImplementedError
