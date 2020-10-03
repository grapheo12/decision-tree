from collections import OrderedDict


class DecisionTreeNode:
    """
    Base Data Structure representing a node of Decision Tree
    ----
    Attributes:

    idx: Unique id of the node

    discriminator: The function forming the basis of splitting at this node
    On a data example x, discriminator(x) returns the edge to be followed
    to reach a target node.

    discriminator_text: Help text describing the discriminator

    height: The level at which the node is present.

    children: A dict mapping edge values as returned by discriminator
    to the exact the child nodes.

    isLeaf: Boolean describing whether the current node is a leaf.

    decision: If the node is leaf, what is the decision on the test example.

    majority: For a non-leaf node, what is the decision
    going by majority training examples at that node.
    """
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
        """
        Add a new child node
        ---
        Params:

        key: The edge value used to reach to this child.
        node: An instance of DecisionTreeNode to be added as the child.
        replace: Replace an existing node, or raise error if node exists.
        """
        if not replace and key in self.children:
            raise Exception("Child with same key exists")
        if node is not None:
            self.children[key] = node

    def descend(self, arg):
        """
        Move to a child from current node.
        ---
        Param:
        arg: The test example under consideration

        Returns:
        If child exists, returns the child
        else returns None.
        """
        value = self.discriminator(arg)
        if value in self.children:
            return self.children[value]
        else:
            return None

    def preorderTraverse(self, parent=None, parentRelation=None):
        """
        Generator returning the Decision Tree nodes
        in root-then-children order.
        ---
        Params:
        parent, parentRelation are internal.

        Returns:
        On each yield,
        returns a tuple (parent, relation to parent, current node)
        """
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
