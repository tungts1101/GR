import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from steinertree import Tree
import random
from collections import defaultdict, Counter
from functools import reduce,cmp_to_key
from itertools import permutations
import time
from multiprocessing import Process,Manager

class Particle:
    def __init__(self, network):
        edges = network.find_edges()
        random.shuffle(edges)
        t = Tree(root=network.sink,compulsory_nodes=network.terminals)
        t.kruskal(edges,network.N)
        
        self.dom_set = []
        self.dom_count = 0
        self.rank = None
        self.dist = None
        self.pos = t
        self.objective = None

    def __repr__(self):
        return "{}\n".format(self.pos.tree)

class Swarm:
    def __init__(self,network,swarm_size=50,genarations=150,stall_gen=20,m=4,delta=0.01,Pm=0.05,Pc=0.8):
        self.network = network
        self.swarm_size = swarm_size
        self.generations = genarations
        self.stall_gen = stall_gen
        self.delta = delta
        
        # objective functions
        self.obj_func = (
            self.network.energy_consumption,
            self.network.network_lifetime,
            self.network.convergence_time,
            self.network.communication_interference
        )
        #####
        
        # global value
        self.cur_gen = 1
        self.cur_sg = 0
        self.g_err = 1e9
        self.g_pos = None
        self.running_time = 0
        self.zmax = [1e-9] * m
        self.zmin = [1e9] * m

    def __repr__(self):
        return "g_err: {}\n".format(self.g_err)
    
    def update_global(self):
        for particle in self.swarm:
            fv = self.fitness_evaluation(particle)

            if fv < self.g_err:
                self.g_pos = particle.pos
                self.g_err = float(fv)
    
    def calculate_objective(self,p):
        return (
            self.network.energy_consumption(p.pos),
            self.network.network_lifetime(p.pos),
            self.network.convergence_time(p.pos),
            self.network.communication_interference(p.pos)
        )
    
    def update_objective(self,objective):
        self.zmax[:] = [max(x,y) for (x,y) in zip(self.zmax,objective)]
        self.zmin[:] = [min(x,y) for (x,y) in zip(self.zmin,objective)]

    # check if pos is dominated
    def is_dominated(self,p1):
        for p2 in self.swarm:
            if self.dominates(p2,p1):
                return True

        return False

    def dominates(self,p1,p2):
        return reduce((lambda x,y:x&y),[x<y for (x,y) in zip(p1.objective,p2.objective)])

    def fitness_evaluation(self,particle):
        objective = particle.objective
        weighted_sum = sum([(objective[j] - self.zmin[j] + 1e-6)/(self.zmax[j] - self.zmin[j] + 1e-6) for j in range(4)])
        pareto = 1 if self.is_dominated(particle) else 0
        return weighted_sum + pareto
    
    def can_stop(self):
        return self.cur_gen > self.generations or self.cur_sg > self.stall_gen or self.running_time > 15*60000
    
    def eval(self):
        def fast_nondominated_sort(pop):
            fronts = [[]]
            
            for p1 in pop:
                p1.dom_count = 0; p1.dom_set = []
                
                for p2 in pop:
                    if self.dominates(p1,p2):
                        p1.dom_set.append(p2)
                    elif self.dominates(p2,p1):
                        p1.dom_count += 1

                if p1.dom_count == 0:
                    p1.rank = 0
                    fronts[0].append(p1)

            curr = 0
            while curr < len(fronts):
                next_front = []
                for p1 in fronts[curr]:
                    for p2 in p1.dom_set:
                        p2.dom_count -= 1
                        if p2.dom_count == 0:
                            p2.rank = curr+1
                            next_front.append(p2)
                curr += 1
                if next_front:
                    fronts.append(next_front)

            return fronts

        def calculate_crowding_distance(pop):
            for p in pop:
                p.dist = 0

            zmax = [1e-9]*4
            zmin = [1e9]*4
            for i in range(4):
                for p in pop:
                    zmax = [max(x,y) for (x,y) in zip(zmax,p.objective)]
                    zmin = [min(x,y) for (x,y) in zip(zmin,p.objective)]
                rge = zmax[i] - zmin[i]
                pop[0].dist, pop[len(pop)-1].dist = 1e15, 1e15
                if rge:
                    for j in range(1,len(pop)-1):
                        pop[j].dist += (pop[j+1].objective[i] - pop[j-1].objective[i])/rge
    
        def crowd_comparison_operator(x,y):        
            return x.dist <= y.dist if x.rank == y.rank else x.rank <= y.rank

        def select_parents(fronts, pop_size):
            for f in fronts:
                calculate_crowding_distance(f)
            offspring, last_front = [], 0
            for front in fronts:
                if len(offspring) + len(front) < pop_size:
                    for p in front:
                        offspring.append(p)
                    last_front += 1
            
            remaining = pop_size - len(offspring)
            if remaining > 0:
                fronts[last_front].sort(key=cmp_to_key(crowd_comparison_operator))
                offspring += fronts[last_front][:remaining+1]

            return offspring
        
        def crossover(p1,p2):
            p = Particle(self.network)

            t1 = p1.pos
            t2 = p2.pos
            t1_fringe = t1.find_fringe()
            t2_fringe = t2.find_fringe()

            intersections = [(x,y) for (x,y) in t1_fringe if (x,y) in t2_fringe]
            unions = list(set().union(t1_fringe,t2_fringe))
            
            unions.sort(key=lambda x:self.network.distance(x[0],x[1]))

            t = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
            t.kruskal(intersections + unions,self.network.N)

            p.pos = t
            return p
        
        def mutate(particle):
            p = Particle(self.network)

            tree = particle.pos

            begin,end = random.sample(tree.find_nodes(),2)
            while self.network.distance(begin,end) > self.network.trans_range:
                begin,end = random.sample(tree.find_nodes(),2)

            def adj(tree,node):
                return [x if y is node else y for (x,y) in tree.find_fringe() if node in (x,y)]

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

                    for n in adj(tree,node):
                        if n not in visited:
                            new_path = list(path)
                            new_path.append(n)
                            fringes.append(new_path)

            redundant = [(x,y) for (x,y) in tree.find_fringe() if x in path and y in path]

            fringes = list(set(tree.find_fringe()) - set([random.choice(redundant)])) + [(begin,end)]
            t = Tree(root=self.network.sink,compulsory_nodes=self.network.terminals)
            t.kruskal(fringes,self.network.N)
            p.pos = t
            return p

        def crossover_and_mutation(pop,f_crossover,f_mutate):
            Q = []

            while len(Q) < self.swarm_size:
                p1,p2 = random.sample(pop,2)

                c = f_crossover(p1,p2)
                c = f_mutate(c)
                Q.append(c)

            return Q

        def calculate_objectives(pop):
            for p in pop:
                p.objective = self.calculate_objective(p)
                self.update_objective(p.objective)

        def init_population():
            s = [Particle(self.network) for _ in range(self.swarm_size)]
            return s

        self.swarm = init_population()
        children = crossover_and_mutation(self.swarm,crossover,mutate)

        start = time.time()

        while not self.can_stop():
            prv_g_err = float(self.g_err)

            union = self.swarm + children
            calculate_objectives(union)
            fronts = fast_nondominated_sort(union)
            
            self.swarm = select_parents(fronts,self.swarm_size)
            self.update_global()
            children = crossover_and_mutation(self.swarm,crossover,mutate)
            
            cur_g_err = float(self.g_err)
            self.cur_sg = self.cur_sg + 1 if abs(cur_g_err - prv_g_err) < self.delta else 0
            self.cur_gen += 1
            self.running_time = time.time() - start

        if not self.g_pos.is_fisible(self.network.distance,self.network.trans_range):
            print('Not OK')
            exit(0)

        return {"error": self.g_err, "running_time": self.running_time, "generations": self.cur_gen}
