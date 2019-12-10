from steinertree import SteinerTree
import random

class Helper:
    def __init__(self):
        pass
    
    @classmethod
    def init_solution(cls,network):
        st = SteinerTree(network)
        st.random_init()

        return st

    @classmethod
    def crossover(cls,t1,t2,network):
        assert(t1.__class__.__name__ == "SteinerTree")
        assert(t2.__class__.__name__ == "SteinerTree")

        t1_fringe = t1.get_all_edges()
        t2_fringe = t2.get_all_edges()

        intersections = [(x,y) for (x,y) in t1_fringe if (x,y) in t2_fringe]
        unions = list(set().union(t1_fringe,t2_fringe))
        
        unions.sort(key=lambda x:network.distance(x[0],x[1]))

        tree = SteinerTree(network)
        tree.kruskal(intersections + unions)
        return tree

    @classmethod
    def mutate(cls,tree,network):
        begin,end = random.sample(tree.get_all_nodes(),2)
        
        i = 0
        while network.distance(begin,end) > network.trans_range and i < 10:
            begin,end = random.sample(tree.get_all_nodes(),2)
            i += 1

        if i == 10:
            return tree
        else:
            fringes = []
            fringes.append([begin])
            visited = []
            path = []

            while len(fringes) > 0:
                path = fringes.pop(0)
                node = path[-1]
                if node not in visited:
                    visited.append(node)
                    if node is end:
                        break

                    for n in tree.get_all_adjacent_nodes(node):
                        if n not in visited:
                            new_path = list(path)
                            new_path.append(n)
                            fringes.append(new_path)

            redundant = [(x,y) for (x,y) in tree.get_all_edges() if x in path and y in path]

            fringes = list(set(tree.get_all_edges()) - set([random.choice(redundant)])) + [(begin,end)]
            t = SteinerTree(network)
            t.kruskal(fringes)
            return t

    @classmethod
    def encode(self,N):
        layer = []
        stack = []
        stack.append((None,self.root))

        while stack:
            parent,node = stack.pop()
            if parent is not None:
                layer.append(parent)
                layer.append(node)

            edges = zip([node]*len(self.tree[node]),self.tree[node])
            stack.extend([(x,y) for (x,y) in edges])

        while len(layer) is not N:
            layer.append(random.choice(layer))

        return layer

