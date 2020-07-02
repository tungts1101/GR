from steinertree import SteinerTree
from collections import defaultdict as dd
from collections import Counter
import random
import config


# double layer encoding library
class DLE:
    def __init__(self, network):
        self.layer_1 = []
        self.layer_2 = []
        self.network = network

    # encode a steiner tree to 2 layer
    @classmethod
    def encode(cls, st):
        res = DLE(st.network)
        n = st.network.N
        
        # generate layer 2
        stack = [(None, st.root)]

        while stack:
            parent,node = stack.pop()
            if parent is not None:
                res.layer_2.append(parent)
                res.layer_2.append(node)
            edges = zip([node]*len(st.child[node]), st.child[node])
            stack.extend([(x,y) for (x,y) in edges])

        # length of layer 2 equals to 2*(n-1) with n is the total number of nodes
        while len(res.layer_2) != 2*(n-1):
            res.layer_2.append(random.choice(res.layer_2))

        # generate layer 1
        for i in range(n):
            if i not in st.terminals:
                if i in set(res.layer_2):
                    res.layer_1.append(1)
                else:
                    res.layer_1.append(0)
        
        return res

    # repair layer 1 by set steiner nodes to 1 and others to 0
    def repair(self):
        self.layer_1 = []

        for i in range(self.network.N):
            if i not in self.network.terminals:
                if i in set(self.layer_2):
                    self.layer_1.append(1)
                else:
                    self.layer_1.append(0)

    def decode(self):
        def find_root(root, node):
            if root[node] == node:
                return node
            return find_root(root,root[node])
        
        def union(root,rank,x,y):
            x_root = find_root(root,x)
            y_root = find_root(root,y)
            if rank[x_root] < rank[y_root]:
                root[x_root] = y_root
            elif rank[y_root] < rank[x_root]:
                root[y_root] = x_root
            else:
                root[y_root] = x_root
                rank[x_root] += 1

        tree = SteinerTree(self.network)
        
        fringes = []
        root = [None for _ in range(len(self.layer_2)//2+1)]
        rank = [0 for _ in range(len(self.layer_2)//2+1)]

        for node in set(self.layer_2):
            root[node] = node

        for i in range(len(self.layer_2)-1):
            x,y = self.layer_2[i],self.layer_2[i+1]

            if self.network.distance(x,y) <= self.network.trans_range:
                x_root = find_root(root,x)
                y_root = find_root(root,y)

                if x_root != y_root:
                    fringes.append((x,y))
                    union(root,rank,x_root,y_root)

        tree.build(fringes)

        return tree

    def is_compatible(self):
        layer = []
        
        for node in range(self.network.N):
            if node not in self.network.terminals:
                if node in set(self.layer_2):
                    layer.append(1)
                else:
                    layer.append(0)
        
        return self.layer_1 == layer

    def fitness_evaluation(self):
        st = self.decode()
        
        return {
            'ec': self.network.energy_consumption(st),
            'nl': self.network.network_lifetime(st),
            'ct': self.network.convergence_time(st),
            'ci': self.network.communication_interference(st)
        }
    
    @classmethod
    def ewd_evolution(cls, p1, p2):
        # get list of adjacent nodes for each node in tree
        def adj_map(st):
            m = dd(lambda: [])
            for node in st.get_all_nodes():
                p = st.parent[node]
                if p is not None:
                    m[node].append(p)
                m[node].extend(st.child[node])
            return m

        # select node with the smallest distance to the current node
        def find_node(node, candidates, network):
            return min(candidates, key=lambda next_node: network.distance(next_node, node))

        # adjacent node crossover
        def anx(p1, p2):
            network = p1.network

            st1 = p1.decode()
            st2 = p2.decode()

            # build up 2 adjacent map
            p1_map = adj_map(st1)
            p2_map = adj_map(st2)
            
            layer = [None for _ in range(len(p1.layer_2))]

            # first node is the root of tree 1
            node = st1.root
            layer[0] = node

            for i in range(1, len(layer)):
                # list of nodes are proximity with current node in BOTH trees
                s = [x for x in p1_map[node] if x in set(p2_map[node])]
                # list of nodes are proximity with current node in AT LEAST one tree
                v = list(set().union(p1_map[node], p2_map[node]))

                if len(s) > 0:
                    next_node = find_node(node, s, network)

                    p1_map[node].remove(next_node)
                    p1_map[next_node].remove(node)

                    p2_map[node].remove(next_node)
                    p2_map[next_node].remove(node)
                    
                    node = next_node
                else:
                    if len(v) > 0:
                        next_node = find_node(node, v, network)

                        if next_node in p1_map[node]:
                            p1_map[node].remove(next_node)
                            p1_map[next_node].remove(node)
                        else:
                            p2_map[node].remove(next_node)
                            p2_map[next_node].remove(node)
                        
                        node = next_node
                    else:
                        node = random.choice(st1.get_all_nodes())

                layer[i] = node

            if config.HARD_FIX:
                # make sure that all nodes in p1 will be in layer
                for node in set(p1.layer_2):
                    if node not in set(layer):
                        counter = Counter(layer)
                        r = random.choice([x for x in counter if counter[x] > 1])
                        layer[layer.index(r)] = node

            return layer

        # reciprocal exchange mutation
        # exchange value of 2 nodes in layer
        def rem(layer):
            i1, i2 = random.sample(range(len(layer)), 2)
            layer[i1], layer[i2] = layer[i2], layer[i1]

        offspring = anx(p1, p2)
        rem(offspring)

        res = DLE(p1.network)
        res.layer_1 = list(p1.layer_1)
        res.layer_2 = offspring

        return res
    
    @classmethod
    def tree_evolution(cls, p1, p2):

        # partial OR operator
        # i1 = 2, i2 = 5
        # p1: 1 0 | 0 1 0 | 0 1
        # p2: 0 1 | 1 1 0 | 1 1
        # partial: (0 1 0) | (1 1 0)
        # offspring: 1 0 1 1 0 0 1
        def poo(_p1, _p2):
            if config.DEBUG:
                assert len(_p1.layer_1) == len(_p2.layer_1)

            i1, i2 = sorted(random.sample(range(len(_p1.layer_1)), 2))
            partial = []
            for i in range(i1, i2):
                partial.append(_p1.layer_1[i] | _p2.layer_1[i])

            offspring = _p1.layer_1[:i1] + partial + _p1.layer_1[i2:]
            return offspring
        
        layer_1 = poo(p1,p2)

        st1 = p1.decode()
        st2 = p2.decode()

        # first priority: edges appear in both trees
        intersections = [(x,y) for (x,y) in st1.get_all_edges() if (x,y) in set(st2.get_all_edges())]

        # second priority: edges contain 2 steiner nodes in layer 1
        unions = list(set().union(st1.get_all_edges(),st2.get_all_edges()))
        residuals = [edge for edge in unions if edge not in intersections]
        steiner_nodes = [node for i, node in enumerate(p1.network.relays) if layer_1[i] == 1]
        steiner_edges = [(n1,n2) for (n1,n2) in residuals if n1 in steiner_nodes and n2 in steiner_nodes]

        # third priority: remaining edges in both trees
        remains = [edge for edge in residuals if edge not in steiner_edges]
        remains.sort(key=lambda x: p1.network.distance(x[0],x[1]))

        st = SteinerTree(p1.network)
        st.kruskal(intersections + steiner_edges + remains)

        return DLE.encode(st)
