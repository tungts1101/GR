import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'lib'))

from network_data import Network
from jpso_double import Swarm as Swarm_v1
from jpso_double_v2 import Swarm as Swarm_v2
from steinertree import Tree
import random
from multiprocessing import Process,Value

fp = os.path.join(os.path.dirname(__file__),'../../WusnNewModel/data')

def get_file(d):
    fn = []
    for f in os.listdir(d):
        if f[-2:] == 'in':
            fn.append(d+f)
    return fn

small = get_file(fp+'/small_data/')
medium = get_file(fp+'/medium_data/')

def handle_file(f,swarm,v,t):
    nw = Network(f)
    s = swarm(nw)
    r = s.eval()

    v.value += r['value']
    t.value += r['time']

def cal(fl,swarm): 
    v = Value('d',0.0)
    t = Value('d',0.0)
    
    all_processes = [Process(target=handle_file,args=(f,swarm,v,t)) for f in fl]

    for p in all_processes:
        p.start()

    for p in all_processes:
        p.join()
    
    l = len(fl)
    return (v.value/l,t.value/l)

def draw():
    try:
        v1_s,t1_s = cal(small,Swarm_v1)
        v2_s,t2_s = cal(small,Swarm_v2)

        #v1_m,t1_m = cal(medium,Swarm_v1)
        #v2_m,t2_m = cal(medium,Swarm_v2)
    except Exception as ex:
        print(ex)
        exit(1)

    print('|{:30}|{:^30}|{:^30}|'.format('','Value','Time'))
    print('|{:30}|{:^15}|{:^14}|{:^15}|{:^14}|'.format('','JPSO Double','GA','JPSO Double','GA'))

    us = [None] * 3
    us[0] = ''.join(['_' for _ in range(30)])
    us[1] = ''.join(['_' for _ in range(15)])
    us[2] = ''.join(['_' for _ in range(14)])

    print('|{:30}|{:15}|{:14}|{:15}|{:14}|'.format(us[0],us[1],us[2],us[1],us[2]))
    
    print('|{:30}|{:^15.2f}|{:^14.2f}|{:^15.2f}|{:^14.2f}|'.format('Source & relay nodes = 40',v1_s,v2_s,t1_s,t2_s))
    
    print('{:92}'.format(''.join(['_' for _ in range(92)])))
    
    #print('|{:30}|{:^15.2f}|{:^14.2f}|{:^15.2f}|{:^14.2f}|'.format('Source & relay nodes = 100',v1_m,v2_m,t1_m,t2_m))
    
    #print('{:92}'.format(''.join(['_' for _ in range(92)])))


    exit(0)

draw()
