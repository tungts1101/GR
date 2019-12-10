from collections import defaultdict as dd
import random

class SteinerTree:
    def __init__(self,network):
        self.network = network
        self.root = network.sink
        self.terminals = network.terminals
        self.compulsory_nodes = [node for node in network.terminals if node != self.root]
        self.child = dd(lambda: [])
        self.parent = dd(lambda: None)
        self.leaves = []
    
    def random_init(self):
        edges = self.network.find_edges()
        random.shuffle(edges)
        self.kruskal(edges)

    def kruskal(self,edges):
        def find_root(root, node):
            if root[node] == node:
                return node
            return find_root(root, root[node])

        def union(root, rank, v1, v2):
            if rank[v1] < rank[v2]:
                root[v1] = v2
            elif rank[v2] < rank[v1]:
                root[v2] = v1
            else:
                root[v2] = v1
                rank[v1] += 1

        fringes = []
        root = [_ for _ in range(self.network.N)]
        rank = [0 for _ in range(self.network.N)]

        for v1,v2 in edges:
            v1_root = find_root(root, v1)
            v2_root = find_root(root, v2)

            if v1_root != v2_root:
                fringes.append((v1, v2))
                union(root, rank, v1_root, v2_root)
        
        self.build(fringes)
    
    def clear(self):
        self.child = dd(lambda: [])
        self.parent = dd(lambda: None)

    def build(self,fringes):
        self.clear()
        queue = [self.root]
        visited = set()
        
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                adjacent_nodes = [x if node == y else y for (x,y) in fringes if node in (x,y)]
                children = [x for x in adjacent_nodes if x not in visited]
                self.child[node].extend(children)
                
                for child in children:
                    self.parent[child] = node
                queue.extend(children)

            if not self.child[node]:
                self.leaves.append(node)

        while not self.is_leaves_fisible():
            self.trimming()
            
    def is_sublist(self,l1,l2):
        return all(e in l2 for e in l1)
    
    def is_leaves_fisible(self):
        return self.is_sublist(self.leaves, self.compulsory_nodes)

    def trimming(self):
        redundants = set(self.leaves) - set(self.compulsory_nodes)

        for node in redundants:
            self.leaves.remove(node)
            # ???
            if node in self.child[self.parent[node]]:
                self.child[self.parent[node]].remove(node)
            self.parent[node] = None

    def is_all_terminals_contained(self):
        return self.is_sublist(self.terminals, self.get_all_nodes())

    def bfs(self, f, *args):
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            f(node, *args)

            for child in self.child[node]:
                queue.append(child)

    def get_all_nodes(self):
        def f(node, ret):
            ret.append(node)

        ret = []
        self.bfs(f,ret)

        return ret

    def get_all_edges(self):
        def f(node, ret):
            for child in self.child[node]:
                ret.append((node,child))

        ret = []
        self.bfs(f, ret)
        return ret
    
    def is_edge_fisible(self):
        for x,y in self.get_all_edges():
            if self.network.distance(x,y) > self.network.trans_range:
                return False

        return True
    
    def get_all_adjacent_nodes(self,node):
        return [x for x in self.child[node]] + [self.parent[node]]
    
    def is_fisible(self):
        # uncomment below to debug
        if not self.is_leaves_fisible():
            #print("Leaves are not fisible")
            return False
        elif not self.is_all_terminals_contained():
            #print("Not all terminals are being contained")
            return False
        elif not self.is_edge_fisible():
            #print("Not all edges are fisible")
            return False
        else:
            return True
