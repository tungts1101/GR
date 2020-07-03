import random
import time
from helper import Helper
import config


class Particle:
    def __init__(self, network):
        self.pos = Helper.init_solution(network)
        self.objective = None

    def __repr__(self):
        return "{}\n".format(self.pos.child)


class Swarm:
    def __init__(self, network, swarm_size=50, generation=150, stall_gen=20, Pc=0.8, Pm=0.05, delta=0.01):
        self.network = network
        self.swarm_size = swarm_size
        self.generation = generation
        self.Pc = Pc    # probability of crossover
        self.Pm = Pm    # probability of mutate
        self.delta = delta
        self.stall_gen = stall_gen
        self.zmax = [1e-9] * config.NUM_OBJECTIVE
        self.zmin = [1e9] * config.NUM_OBJECTIVE
        self.g_err = 1e9
        self.g_pos = None
        self.cur_gen = 1
        self.cur_sg = 0
        
        self.swarm = []
        for _ in range(self.swarm_size):
            self.swarm.append(Particle(self.network))

    def calculate_objectives(self, pop):
        for p in pop:
            p.objective = self.calculate_objective(p)
            self.update_objective(p.objective)

    def update_global(self):
        for particle in self.swarm:
            err = self.fitness_evaluation(particle)

            if err < self.g_err:
                self.g_pos = particle.pos
                self.g_err = float(err)
    
    def calculate_objective(self, p):
        return (
            self.network.energy_consumption(p.pos),
            self.network.network_lifetime(p.pos),
            self.network.convergence_time(p.pos),
            self.network.communication_interference(p.pos)
        )
    
    def update_objective(self, objective):
        # update maximal and minimum of each objective
        self.zmax[:] = [max(x, y) for (x, y) in zip(self.zmax, objective)]
        self.zmin[:] = [min(x, y) for (x, y) in zip(self.zmin, objective)]

    def is_dominated(self,p):
        for _p in self.swarm:
            if Helper.dominate(_p, p):
                return True

        return False

    def fitness_evaluation(self,particle):
        objective = particle.objective
        weighted_sum = \
            sum([(objective[j] - self.zmin[j] + 1e-6)/max(self.zmax[j] - self.zmin[j], 1e-6)
                 for j in range(config.NUM_OBJECTIVE)])

        pareto = 1 if self.is_dominated(particle) else 0
        return weighted_sum + pareto
    
    # def select(self,q,r):
    #     if r < q[0]:
    #         return 0
    #     for i in range(1,len(q)):
    #         if q[i-1] < r <= q[i]:
    #             return i

    def can_stop(self):
        return self.cur_gen >= self.generation or self.cur_sg > self.stall_gen
    
    def crossover(self, p1, p2):
        p = Particle(self.network)
        p.pos = Helper.crossover(p1.pos, p2.pos, self.network)
        if config.DEBUG:
            assert(p.pos.is_feasible())
        return p
    
    def mutate(self, particle):
        p = Particle(self.network)
        p.pos = Helper.mutate(particle.pos, self.network)
        if config.DEBUG:
            assert(p.pos.is_feasible())
        return p

    def eval(self):
        start = time.time()

        if config.TRACE:
            print("Start GA with swarm_size = {0}, max generation = {1}"
                  .format(self.swarm_size, self.generation))

        while not self.can_stop():
            if config.TRACE:
                print("Generation = {0}, Error = {1}".format(self.cur_gen, self.g_err))

            prv_g_err = float(self.g_err)

            P = list(self.swarm)

            # crossover
            c = [i for i in range(len(P)) if random.uniform(0, 1) < self.Pc]
            random.shuffle(c)
            c = c[:len(c) - len(c) % 2]
            Q1 = [self.crossover(P[i], P[j]) for (i, j) in zip(c[:len(c) // 2], c[len(c) // 2:])]

            # mutate
            Q2 = [self.mutate(P[i]) for i in range(len(P)) if random.uniform(0, 1) < self.Pm]

            # selection
            swarm = P + Q1 + Q2
            self.calculate_objectives(swarm)

            # choose |swarm_size| particles with best fitness value
            self.swarm = sorted(swarm, key=lambda particle: self.fitness_evaluation(particle))[:self.swarm_size]

            # update value of global best
            self.update_global()

            cur_g_err = float(self.g_err)
            self.cur_sg = self.cur_sg + 1 if abs(cur_g_err - prv_g_err) < self.delta else 0
            self.cur_gen += 1

        if config.DEBUG:
            assert(self.g_pos.is_feasible())
        running_time = time.time() - start

        if config.TRACE:
            print("End GA at generation = {0} with error = {1} and running time = {2}"
                  .format(self.cur_gen, self.g_err, running_time))
            print("--------------------------------------------------------------------------------")
            print("")

        return {"error": self.g_err, "running_time": running_time, "generation": self.cur_gen}