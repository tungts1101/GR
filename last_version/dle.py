from steinertree import SteinerTree
from collections import defaultdict as dd
from collections import Counter
import random

class DLE:
    def __init__(self, network):
        self.layer_1 = []
        self.layer_2 = []
        self.network = network
    
    @classmethod
    def encode(cls, st):
        res = DLE(st.network)
        n = st.network.N
        
        # generate layer 2
        stack = []
        stack.append((None,st.root))

        while stack:
            parent,node = stack.pop()
            if parent is not None:
                res.layer_2.append(parent)
                res.layer_2.append(node)
            edges = zip([node]*len(st.child[node]),st.child[node])
            stack.extend([(x,y) for (x,y) in edges])
        
        while len(res.layer_2) != 2*(n-1):
            res.layer_2.append(random.choice(res.layer_2))
        
        assert(len(res.layer_2) == 2*(n-1))

        # generate layer 1
        for i in range(n):
            if i not in st.terminals:
                if i in set(res.layer_2):
                    res.layer_1.append(1)
                else:
                    res.layer_1.append(0)
        
        return res
    
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

        res = SteinerTree(self.network)
        
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

        res.build(fringes)

        return res

    def is_compatible(self):
        layer = []
        
        for node in range(self.network.N):
            if node not in self.network.terminals:
                if node in set(self.layer_2):
                    layer.append(1)
                else:
                    layer.append(0)
        
        return (self.layer_1 == layer)

    def fitness_evaluation(self):
        st = self.decode()
        
        return {
            'ec' : self.network.energy_consumption(st),
            'nl' : self.network.network_lifetime(st),
            'ct' : self.network.convergence_time(st),
            'ci' : self.network.communication_interference(st)
        }
    
    @classmethod
    def ewd_evolution(cls,p1,p2):
        def adj_map(st):
            m = dd(lambda: [])
            for node in st.get_all_nodes():
                p = st.parent[node]
                if p is not None: m[node].append(p)
                m[node].extend(st.child[node])
            return m
        
        def find_node(node,candidates,network):
            return min(candidates, key=lambda next_node: network.distance(next_node,node))

        def anx(p1,p2):
            network = p1.network

            st1 = p1.decode()
            st2 = p2.decode()

            p1_map = adj_map(st1)
            p2_map = adj_map(st2)
            
            layer = [None for _ in range(len(p1.layer_2))]

            node = st1.root
            layer[0] = node

            for i in range(1,len(layer)):
                s = [x for x in p1_map[node] if x in set(p2_map[node])]
                v = list(set().union(p1_map[node],p2_map[node]))

                if len(s) > 0:
                    next_node = find_node(node,s,network)

                    p1_map[node].remove(next_node)
                    p1_map[next_node].remove(node)

                    p2_map[node].remove(next_node)
                    p2_map[next_node].remove(node)
                    
                    node = next_node
                else:
                    if len(v) > 0:
                        next_node = find_node(node,v,network)

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
            
            # make sure that all node in p1 will be in layer
            for node in set(p1.layer_2):
                if node not in set(layer):
                    counter = Counter(layer)
                    r = random.choice([x for x in counter if counter[x] > 1])
                    layer[layer.index(r)] = node

            return layer
                          
        def rem(layer):
            i1,i2 = random.sample(range(len(layer)),2)
            layer[i1],layer[i2] = layer[i2],layer[i1]

        offspring = anx(p1,p2)
        rem(offspring)

        res = DLE(p1.network)
        res.layer_1 = list(p1.layer_1)
        res.layer_2 = offspring

        if res.is_compatible() and res.decode().is_fisible():
            return res
        else:
            return p1
    
    @classmethod
    def tree_evolution(cls,p1,p2):
        def poo(p1,p2):
            i1,i2 = sorted(random.sample(range(len(p1.layer_1)),2))
            offspring = p1.layer_1[:i1] + p2.layer_1[i1:i2] + p1.layer_1[i2:]
            return offspring
        
        layer_1 = poo(p1,p2)

        st1 = p1.decode()
        st2 = p2.decode()

        intersections = [(x,y) for (x,y) in st1.get_all_edges() if (x,y) in set(st2.get_all_edges())]
        unions = list(set().union(st1.get_all_edges(),st2.get_all_edges()))
        residuals = [edge for edge in unions if edge not in intersections]
        steiner_nodes = [node for i, node in enumerate(p1.network.relays) if layer_1[i] == 1]
        edges = [[src,dst] for [src,dst] in residuals if src in steiner_nodes and dst in steiner_nodes]

        candidates = [edge for edge in residuals if edge not in edges]
        candidates.sort(key=lambda x:p1.network.distance(x[0],x[1]))

        st = SteinerTree(p1.network)
        st.kruskal(intersections + edges + candidates)

        return DLE.encode(st)
