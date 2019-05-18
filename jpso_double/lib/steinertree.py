from collections import defaultdict
import random

class Tree:
    def __init__(self,root,compulsory_nodes):
        self.root = root
        self.compulsory_nodes = compulsory_nodes
        self.tree = defaultdict(lambda: [])
    
    # helper function for kruskal algorithm
    def __find_parent(self,parent,i):
        if parent[i] == i:
            return i
        return self.__find_parent(parent,parent[i])
    
    # helper function for kruskal algorithm
    def __union(self,parent,rank,x,y):
        xroot = self.__find_parent(parent,x)
        yroot = self.__find_parent(parent,y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[yroot] < rank[xroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
    
    # helper function to check if list2 contains all elements in list1
    def __is_sublist(self,list1,list2):
        return all(elem in list2 for elem in list1)

    #####
    # algorithm derived from kruskal to construct a steiner tree
    # input:
    # - edges: edge set to construct
    # - V: max vertices can be included in tree
    # output: a steiner tree
    #####
    def kruskal(self,edges,V):
        def can_stop():
            nodes = set([node for edge in fringe for node in edge])
            return self.__is_sublist(self.compulsory_nodes,nodes) and len(nodes) == len(fringe) + 1
        
        fringe = []
        parent = [None] * V
        rank = [0] * V

        for vertex in range(V):
            parent[vertex] = vertex
        
        for x,y in edges:
            xroot = self.__find_parent(parent,x)
            yroot = self.__find_parent(parent,y)

            if xroot != yroot:
                fringe.append((x,y))
                self.__union(parent,rank,xroot,yroot)
                #if can_stop():
                #    break
        
        #print(list(set(node for edge in fringe for node in edge)))
        self.__construct_tree(fringe)
        
    #####
    # construct a fisible tree from list of fringes
    # input:
    # - fringe: list of fringe to be included
    #####
    def __construct_tree(self,fringe):
        # remember to clear tree
        self.tree = defaultdict(lambda: [])
        queue = []
        visited = []
        queue.append(self.root)

        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                adjacent_nodes = [x if node == y else y for (x,y) in fringe if node in (x,y)]
                children = [x for x in adjacent_nodes if x not in visited]
                self.tree[node].extend(children)
                queue.extend(children)

        # apply trimming strategy to remove all leaves which are not compulsory
        while not self.__is_fisible():
            self.__trimming()
    
    def __is_fisible(self):
        return self.__is_sublist(self.__leaves(),self.compulsory_nodes)

    # remove all leaf nodes which are not compulsory nodes
    def __trimming(self):
        def find_parent(node):
            for n in self.tree:
                if node in self.tree[n]:
                    return n
            return None

        for n in self.__leaves():
            if n not in self.compulsory_nodes:
                del(self.tree[n])
                self.tree[find_parent(n)].remove(n)

    # return all leaves
    def __leaves(self):
        return [node for node in self.tree if len(self.tree[node]) is 0]

    #####
    # decode function: construct a tree from a layer
    # using cf-tcr (cycle free tree coding routine)
    # a Prim derived algorithm from
    # EWD (edge window decoder) strategy
    # input:
    # - a layer
    # output: a steiner tree
    #####
    def decode(self,layer,cal_dis,trans_range):
        fringe = []
        parent = [None]*(len(layer)//2+1)
        rank = [0]*(len(layer)//2+1)

        for node in set(layer):
            parent[node] = node
        
        for x,y in zip(layer[0:-1],layer[1:]):
            if cal_dis(x,y) <= trans_range:
                xroot = self.__find_parent(parent,x)
                yroot = self.__find_parent(parent,y)

                if xroot != yroot:
                    fringe.append((x,y))
                    self.__union(parent,rank,xroot,yroot)

        self.__construct_tree(fringe)

    #####
    # encode function: create a layer length N
    # input: a required length of output layer
    # output: a layer has length N
    #####
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

    # return all descendants of a node
    def find_des(self,node):
        des = []
        queue = []
        queue.append(node)

        while queue:
            node = queue.pop(0)
            des.extend(self.tree[node])
            queue.extend(self.tree[node])

        return des

    # return all fringe in tree
    def find_fringe(self):
        fringe = []

        for node in self.tree:
            for child in self.tree[node]:
                fringe.append((node,child))

        return fringe

    # return parent of a node
    def find_parent(self,node):
        for n in self.tree:
            if node in self.tree[n]:
                return n
        
        # in case node is root
        return None

    # return all nodes in tree
    def find_nodes(self):
        nodes = []
        for node in self.tree:
            if node not in nodes: nodes.append(node)
            for child in self.tree[node]:
                if child not in nodes: nodes.append(child)

        return nodes
    
    # check if tree is a fisible solution
    def is_fisible(self,cal_dis,trans_range):
        if not self.__is_fisible():
            #print('leaf not good')
            return False
        
        if not self.__is_sublist(self.compulsory_nodes,self.find_nodes()):
            #print('not contain all nodes')
            return False

        for x,y in self.find_fringe():
            if cal_dis(x,y) > trans_range:
                #print('edge not fisible')
                return False

        return True
