from dtree.tree import DecisionTreeNode


def test_tree():
    root = DecisionTreeNode(idx=1, discriminator=lambda x: x > 0,
                            discriminator_text="x > 0", height=0)

    node1 = DecisionTreeNode(idx=2, discriminator=lambda y: y <= 0,
                             discriminator_text="y <= 0", height=1)
    node2 = DecisionTreeNode(idx=3, discriminator=lambda y: y > 0,
                             discriminator_text="y > 0", height=1)
    node3 = DecisionTreeNode(idx=4, discriminator=lambda z: z < 0,
                             discriminator_text="y <= 0", height=2)
    node4 = DecisionTreeNode(idx=5, discriminator=lambda z: z > 0,
                             discriminator_text="z > 0", height=2)

    root.addChild(key=True, node=node1)
    root.addChild(key=False, node=node2)
    node2.addChild(key=True, node=node3)
    node1.addChild(key=True, node=node4)

    for parent, relation, node in root.preorderTraverse():
        if parent is not None:
            print(parent.idx, relation, node.idx, node.discriminator_text)
        else:
            print(None, relation, node.idx, node.discriminator_text)

    data = [50, 0, 30]
    node = root
    i = 0
    while i < len(data) and node is not None:
        print(node.idx, node.discriminator_text)
        node = node.descend(data[i])
        i += 1
