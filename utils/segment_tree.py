import numpy as np

tree = SegmentTree(4)
print(tree.data)
print(tree.sum_nodes)
tree.append(0, 0.1)
tree.append(1, 0.3)
tree.append(2, 0.4)
tree.append(3, 0.2)
tree.find(0.5)
tree.find(0.15)

class SegmentTree(object):
    def __init__(self, num_leaf_nodes):
        # TODO: let's change to shared memory later

        # setup
        self.data_ind = 0
        self.full = False

        self.data_size = num_leaf_nodes
        self.data = [None] * self.data_size        # the actual data goes here
        self.sum_nodes = [0] * (self.data_size * 2 - 1) # the sum tree: each node stores the sum of subtrees

        # NOTE: in per and rainbow, the new experiences are added w/ max priority
        # NOTE: whereas in distributed, they are added w/ the priority calculated by individual actors
        # self.max_priority = 1   # init the priority of the most recent samples w/ the max

    def _propagate(self, node_ind, value):
        parent = (node_ind - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_nodes[parent] = self.sum_nodes[left] + self.sum_nodes[right]
        if parent != 0:
            self._propagate(parent, value)

    def _update(self, node_ind, value):
        self.sum_nodes[node_ind] = value
        self._propagate(node_ind, value)
        # self.max_priority = max(value, self.max_priority)

    def append(self, data, value):
        self.data[self.data_ind] = data
        self._update(self.data_size - 1 + self.data_ind, value)
        self.data_ind += 1
        if self.data_ind == self.data_size:
            self.full = True
            self.data_ind = 0

    # searches for the location of a value in sum tree
    def _retrieve(self, node_ind, value):
        left, right = 2 * node_ind + 1, 2 * node_ind + 2
        if left >= len(self.sum_nodes):
            return node_ind
        elif value <= self.sum_nodes[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_nodes[left])

    # searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        node_ind = self._retrieve(0, value)
        data_ind = node_ind - self.data_size + 1
        return (self.sum_nodes[node_ind], data_ind, node_ind)

    def get(self, data_ind):
        return self.data[data_ind % self.data_size]

    def total_sum(self):
        return self.sum_nodes[0]
