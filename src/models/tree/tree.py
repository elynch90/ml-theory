class Node:
    def __init__(self, left=None, right=None, val=None):
        self.left = left
        self.right = right
        self.val = val


def depth_first_search(node):
    if node is None:
        return
    print(node.val)
    depth_first_search(node.left)
    depth_first_search(node.right)


def breadth_first_search(node):
    queue = [node]
    while queue:
        next_level = []
        this_node = queue.pop(0)
        print(this_node.val)
        if this_node.left:
            next_level.append(this_node.left)
        if this_node.right:
            next_level.append(this_node.right)
        queue = next_level
