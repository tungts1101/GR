from steinertree import SteinerTree
import random
from functools import reduce
from dle import DLE
import time
import config


class Particle:
    def __init__(self,network):
        self.network = network

        st = SteinerTree(network)
        # important
        st.random_init()
        self.c_pos = DLE.encode(st)

        res = self.c_pos.fitness_evaluation()
        self.objective = [res['ec'],res['nl'],res['ct'],res['ci']]

        self.b_pos = self.c_pos
        self.c_err, self.b_err = None, None
    
    def fly(self,pos,flag):
        if flag:
            offspring = DLE.ewd_evolution(self.c_pos, pos)
        else:
            offspring = DLE.tree_evolution(self.c_pos,pos)

        return offspring
    
    def repair(self):
        self.c_pos.repair()

    def eval(self):
        res = self.c_pos.fitness_evaluation()
        self.objective = [res['ec'],res['nl'],res['ct'],res['ci']]

    def update(self,zmax,zmin,flag):
        self.c_err = self.fv(zmax,zmin,flag)

        if self.b_err == None or self.c_err < self.b_err:
            self.b_pos = self.c_pos
            self.b_err = float(self.c_err)

    def fv(self,zmax,zmin,flag):
        weighted_sum = sum([(self.objective[j] - zmin[j] + 1e-6)/(zmax[j]-zmin[j] + 1e-6) for j in range(len(self.objective))])
        pareto = 1 if flag else 0
        return weighted_sum + pareto

class Swarm:
    def __init__(self,network,swarm_size=50,generations=150,stall_gen=20,delta=0.01,C0=0,C1=0.5):
        self.network = network
        self.swarm_size = swarm_size
        self.generations = generations
        self.stall_gen = stall_gen
        self.delta = delta
        self.C0 = C0
        self.C1 = C1

        self.g_pos = []
        self.g_err = 1e9
        self.zmin = [1e9 for _ in range(4)]
        self.zmax = [1e-9 for _ in range(4)]
        self.cur_gen = 1
        self.cur_sg = 0

        self.swarm = []
   
    def init_swarm(self):
        for _ in range(self.swarm_size):
            p = Particle(self.network)
            p.eval()
            self.update_objective(p.objective)
            self.swarm.append(p)

        for p in self.swarm:
            flag = self.is_dominated(p)
            p.update(self.zmax,self.zmin,flag)

        self.update_global()

    def is_dominated(self, p):
        for _p in self.swarm:
            if self.dominates(_p,p):
                return True
        return False

    def dominates(self,p1,p2):
        return reduce((lambda x,y:x&y),[x<y for (x,y) in zip(p1.objective,p2.objective)])
    
    def update_objective(self,objective):
        self.zmax[:] = [max(x,y) for (x,y) in zip(self.zmax,objective)]
        self.zmin[:] = [min(x,y) for (x,y) in zip(self.zmin,objective)]

    def update_global(self):
        for particle in self.swarm:
            if particle.b_err < self.g_err:
                self.g_pos = particle.b_pos
                self.g_err = float(particle.b_err)

    def can_stop(self):
        return self.cur_gen > self.generations or self.cur_sg > self.stall_gen

    def eval(self):
        self.init_swarm()

        start = time.time()

        while not self.can_stop():
            prv_g_err = float(self.g_err)
            
            for particle in self.swarm:
                R = random.randint(0,1)

                if self.C0 < R <= self.C0 + self.C1:
                    p_layer_1 = particle.c_pos.layer_1
                    b_layer_1 = particle.b_pos.layer_1
                    flag = (p_layer_1 == b_layer_1)
                    particle.c_pos = particle.fly(particle.b_pos, flag)
                else:
                    p_layer_1 = particle.c_pos.layer_1
                    g_layer_1 = self.g_pos.layer_1
                    flag = (p_layer_1 == g_layer_1)
                    particle.c_pos = particle.fly(self.g_pos, flag)
                
                # repair particle
                particle.repair()

                # update current objective in particle
                particle.eval()

                # update Zmin, Zmax
                self.update_objective(particle.objective)
                
                # update individual best
                flag = self.is_dominated(particle)
                particle.update(self.zmax, self.zmin, flag)

            # update global best
            self.update_global()

            cur_g_err = float(self.g_err)
            self.cur_sg = self.cur_sg + 1 if abs(cur_g_err - prv_g_err) < self.delta else 0
            self.cur_gen += 1

        if config.DEBUG:
            assert(self.g_pos.decode().is_fisible())
        
        running_time = time.time() - start
        return {"error": self.g_err, "running time": running_time, "generation": self.cur_gen}
