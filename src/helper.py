from steinertree import SteinerTree
import random
import copy
import config
from functools import reduce


class Helper:
    def __init__(self):
        pass

    @classmethod
    def init_solution(cls, network):
        st = SteinerTree(network)
        st.random_init()

        return st

    @classmethod
    def crossover(cls, t1, t2, network):
        t1_fringe = t1.get_all_edges()
        t2_fringe = t2.get_all_edges()

        intersections = [(x, y) for (x, y) in t1_fringe if (x, y) in t2_fringe]
        unions = list(set().union(t1_fringe, t2_fringe))

        unions.sort(key=lambda x: network.distance(x[0], x[1]))

        tree = SteinerTree(network)
        tree.kruskal(intersections + unions)
        return tree

    @classmethod
    def mutate(cls, tree, network):
        node1, node2 = random.sample(tree.get_all_nodes(), 2)

        # choose two random nodes in tree
        i = 0
        while network.distance(node1, node2) > network.trans_range and i < 10:
            node1, node2 = random.sample(tree.get_all_nodes(), 2)
            i += 1

            # if we cannot get 2 node in transmission range, return immediately
            if i == 10:
                return tree

        # create a new tree by using shallow copy
        new_tree = copy.copy(tree)

        d1 = new_tree.find_depth(node1)
        d2 = new_tree.find_depth(node2)

        if config.DEBUG:
            print("Before mutate ----------------------------------------------")
            print("d1 = {0}, d2 = {1}".format(d1, d2))
            print("Node 1 = {0}, Node 1 parent = {1}".format(node1, new_tree.parent[node1]))
            print("Node 2 = {0}, Node 2 parent = {1}".format(node2, new_tree.parent[node2]))

        if d1 < d2:
            # set node1 become new parent of node2
            node2_old_parent = new_tree.parent[node2]
            new_tree.child[node2_old_parent].remove(node2)

            new_tree.parent[node2] = node1
            new_tree.child[node1].append(node2)
        else:
            # set node2 become new parent of node1
            node1_old_parent = new_tree.parent[node1]
            new_tree.child[node1_old_parent].remove(node1)

            new_tree.parent[node1] = node2
            new_tree.child[node2].append(node1)

        if config.DEBUG:
            print("After mutate -----------------------------------------------")
            print("Node 1 = {0}, Node 1 parent = {1}".format(node1, new_tree.parent[node1]))
            print("Node 2 = {0}, Node 2 parent = {1}".format(node2, new_tree.parent[node2]))
            print("------------------------------------------------------------")
            print("")

        return new_tree

    @classmethod
    def dominate(cls, p1, p2):
        return reduce((lambda x, y: x & y), [x <= y for (x, y) in zip(p1.objective, p2.objective)])

    # @classmethod
    # def encode(self, N):
    #     layer = []
    #     stack = [(None, self.root)]
    #
    #     while stack:
    #         parent, node = stack.pop()
    #         if parent is not None:
    #             layer.append(parent)
    #             layer.append(node)
    #
    #         edges = zip([node]*len(self.tree[node]),self.tree[node])
    #         stack.extend([(x,y) for (x,y) in edges])
    #
    #     while len(layer) is not N:
    #         layer.append(random.choice(layer))
    #
    #     return layer
