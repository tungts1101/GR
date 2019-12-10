
# heuristic: tim ra ket noi tieu hao nang luong lon nhat
# GA: tim ra ca the co tieu hao nang luong nho nhat

import random, time, os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from .heuristic import *


GEN = 100
CP = 0.8
MP = 0.1
NUM_OF_INDIVIDUALS = 100
TERMINATE = 30
alpha = 0.5

def random_init_individual(num_relay):
    "Initial individual with any num of relay"
    indi = []
    Y = random.randint(1, num_relay)
    xs = Y/num_relay
    count_relay = 0
    for i in range(0, num_relay):
        xx = random.random()
        if xx < xs:
            indi.append(1)
        else:
            indi.append(0)
    return indi

def count_current_relay(individual):
    sum = 0
    for g in individual:
        if g == 1:
            sum += 1
    return sum

# mom and dad instead of parent1 and parent2 =)
def cross(mom, dad):
    num_relay = len(mom)
    mid = random.randint(0, num_relay-1)
    child1 = mom[:mid] + dad[mid:]
    child2 = dad[:mid] + mom[mid:]
    return child1, child2


def mutate(original):
    fake = original[:]
    ll = len(fake)
    count1 = 0
    count2 = 0
    id1 = random.randint(0, ll-1)
    while fake[id1] == 0:
        count1 += 1
        id1 = random.randint(0, ll-1)
        if count1 >= 2*ll:
            break
    id2 = random.randint(0, ll-1)
    while fake[id2] == 1:
        count2 += 1
        id2 = random.randint(0, ll-1)
        if count2 >= 2*ll:
            break
    fake[id1], fake[id2] = fake[id2], fake[id1]
    return fake

def normalize_loss(indi):
    if indi[1].loss(alpha) < 0:
        return float("inf")
    else:
        return 10000*indi[1].loss(alpha) + indi[1].total_tranmission_loss()

# def sort(individuals):
#     ll = len(individuals)
#     new_indis = individuals[:]
#     for i in range(len(individuals)):
#         for j in range(i+1, len(individuals)):
#             if new_indis[i][1].loss(alpha) > new_indis[j][1].loss(alpha):
#                 new_indis[i], new_indis[j] = new_indis[j], new_indis[i]
#             elif new_indis[i][1].loss(alpha) == new_indis[j][1].loss(alpha):
#                 if new_indis[i][1].total_tranmission_loss() > new_indis[j][1].total_tranmission_loss():
#                     new_indis[i], new_indis[j] = new_indis[j], new_indis[i]
#     return new_indis

def GA(inp: WusnInput) -> int:
    # Khoi tao quan the
    individuals = []

    # Cac ca the da duoc tinh toan
    calculated = {}

    for i in range (0, NUM_OF_INDIVIDUALS):
        indi = random_init_individual(inp.num_of_relays)
        out = heuristic(inp, indi)
        
        calculated[str(indi)] = out
        individuals.append([indi, out])

    print(individuals[0])
    
    count_stable = 0
    max_c = individuals[0][1].loss(alpha)
    prev_max = individuals[0][1].loss(alpha)

    # Iterate through generations
    for it in range(0, GEN):
        start = time.time()
        none = 0
        not_none = 0
        # Crossover and mutation
        for id1 in range(0, NUM_OF_INDIVIDUALS):
            id2 = 0
            xx = random.random()
            if xx < CP:
                id2 = random.randint(0, NUM_OF_INDIVIDUALS-1)
                while id2 == id1:
                    id2 = random.randint(0, NUM_OF_INDIVIDUALS-1)
                son, daughter = cross(individuals[id1][0], individuals[id2][0])

                if str(son) in calculated:
                    out1 = calculated[str(son)]
                else:
                    s = time.time()
                    out1 = heuristic(inp, son)
                    t = time.time()
                    
                if str(daughter) in calculated:
                    out2 = calculated[str(daughter)]
                else:
                    # s = time.time()
                    out2 = heuristic(inp, daughter)
                    # t = time.time()

                if out1.mapping == {}:
                    none += 1
                else: 
                    not_none += 1
                if out2.mapping == {}:
                    none += 1
                else:
                    not_none += 1

                individuals.append([son, out1])
                individuals.append([daughter, out2])

                xx2 = random.random()
                if xx2 < MP:
                    grand_child1 = mutate(son)
                    grand_child2 = mutate(daughter)
                    m_out1 = heuristic(inp, grand_child1)
                    m_out2 = heuristic(inp, grand_child2)

                    if m_out1.mapping == {}:
                        none += 1
                    else:
                        not_none += 1
                    if m_out2.mapping == {}:
                        none += 1
                    else: 
                        not_none += 1

                    individuals.append([grand_child1, m_out1])
                    individuals.append([grand_child2, m_out2])

        individuals2 = sorted(individuals, key=normalize_loss)
        # individuals2 = sort(individuals)
        individuals = individuals2[:NUM_OF_INDIVIDUALS-1] 
        individuals.append(individuals2[-1])
        if individuals[0][1].loss(alpha) < max_c:
            max_c = individuals[0][1].loss(alpha)
        if individuals[0][1].loss(alpha) == prev_max:
            count_stable += 1
        else:
            count_stable = 0
        if count_stable == TERMINATE:
            print("TERMINATE")
            break
        prev_max = individuals[0][1].loss(alpha)
        end = time.time()
        print("none: %d, not_none: %d" % (none, not_none))
        print("Gen: %d, time: %fs, min: %f %f %f" % (it, end - start, len(individuals[0][1].used_relays), individuals[0][1].loss(alpha), individuals[NUM_OF_INDIVIDUALS-1][1].loss(alpha)))
    # print(max_c)
    return individuals[0]

# Bo sung dieu kien SN va RN co ket noi duoc voi nhau hay khong bang cach them ban kinh sn va rn
