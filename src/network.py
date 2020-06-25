import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../WusnNewModel'))

from common.input import WusnInput as inp
from collections import defaultdict

time_slot = 1.5
Ee = 50 * 1e-9
k = 4000
Erx = k * Ee
Efs = 100 * 1e-9
Emp = 50 * 1e-12
Eda = 5 * 1e-9
E = 50
d0 = (Efs / Emp) ** 0.5

class Network:
    def __init__(self,filename):
        def coord(point):
            return (point.to_dict()['x'],point.to_dict()['y'],point.to_dict()['z'])

        nw = inp.from_file(filename)
        self.sink = 0
        self.sources = [i for i in range(1,nw.num_of_sensors+1)]
        self.terminals = [self.sink] + self.sources
        self.relays = [i for i in range(nw.num_of_sensors+1,nw.num_of_sensors+nw.num_of_relays+1)]
        self.N = nw.num_of_sensors + nw.num_of_relays + 1
        self.area = (nw.W,nw.H)
        self.trans_range = 2*nw.radius

        self.coord = [coord(p) for p in [nw.BS] + nw.sensors + nw.relays]

    def cal_Etx(self,d):
        if d < d0:
            return Erx + k*Efs*d**2
        else:
            return Erx + k*Emp*d**2**2
    
    def convergence_time(self,st):
        ret = 0
        layer = []
        layer.append(st.root)

        while layer:
            CN = max([len(st.child[sn]) for sn in layer])
            ret += CN * time_slot
            
            layer = [c for sn in layer for c in st.child[sn]]

        return ret

    def distance(self,n1,n2):
        x1,y1,z1 = self.coord[n1]
        x2,y2,z2 = self.coord[n2]
        return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(.5)
    
    def link_interference(self,n1,n2):
        d = self.distance(n1,n2)

        return len([node for node in range(self.N) if (self.distance(node,n1) <= d or self.distance(node,n2) <= d)])
    
    def communication_interference(self,st):
        return max([self.link_interference(n1,n2) for n1,n2 in st.get_all_edges()])
    
    def node_consumption(self,st,node):
        # print("node = {}\n".format(node))
        # return 0 if st.parent[node] is None else 
            # self.cal_Etx(self.distance(node, st.parent[node])) +
           #  sum([len(st.child[node])*(Erx+Eda)])

        # Etx = 0 if st.parent[node] is None else self.cal_Etx(self.distance(node, st.parent[node]))
        return self.cal_Etx(self.distance(node, st.parent[node])) +\
               sum([len(st.child[node])*(Erx+Eda)])
    
    def energy_consumption(self,st):
        return sum([self.node_consumption(st,sn) for sn in st.get_all_nodes() if sn != st.root])

    def network_lifetime(self,st):
        Ec = max([self.node_consumption(st,sn) for sn in st.get_all_nodes() if sn != st.root])
        return Ec/(E - Ec)
    
    def find_edges(self):
        edges = []
        for i in range(self.N):
            for j in range(i+1,self.N):
                if self.distance(i,j) <= self.trans_range:
                    edges.append((i,j))

        return edges

    def adj_dict(self):
        ret = defaultdict(lambda: [])

        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.distance(i,j) <= self.trans_range:
                    ret[i].append(j)

        return ret
