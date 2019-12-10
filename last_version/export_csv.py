import os
import random
from collections import defaultdict
from steinertree import SteinerTree as ST
from network import Network
from ga import Swarm as sga
from pso import Swarm as spso
from multiprocessing import Process,Value,Manager
import csv

fp = os.path.join(os.path.dirname(__file__),'../WusnNewModel/data')

def get_file(d):
    fn = []
    for f in os.listdir(d):
        if f[-2:] == 'in':
            fn.append(d+f)
    return fn

small = get_file(fp+'/small_data/')
medium = get_file(fp+'/medium_data/')

def run_with_same_generations(f,swarm,res):
    nw = Network(f)
    s = swarm(nw,generations=150,stall_gen=150)
    r = s.eval()

    res.append(
        {
            'file' : os.path.split(os.path.abspath(f))[1],
            'error': r['error'],
            'running time' : r['running time']
        }
    )

def run_with_diff_generations(f,swarm,res):
    nw = Network(f)
    s = swarm(nw)
    r = s.eval()

    res.append(
        {
            'file' : os.path.split(os.path.abspath(f))[1],
            'error': r['error'],
            'running time' : r['running time'],
            'generations' : r['generations']
        }
    )


# run each swarm on datasets
def cal(fl,swarm,run_function):
    res = Manager().list()
    
    all_processes = [Process(target=run_function,args=(f,swarm,res)) for f in fl]

    for p in all_processes:
        p.start()

    for p in all_processes:
        p.join()
    
    res = list(res)

    return res

# def mean(arr):
    # return sum(arr)/len(arr)

# def variance(arr):
    # return sum([i**2 for i in arr])/len(arr) - mean(arr)**2

def export_csv(csvf,r1,r2):
    with open(csvf,mode='w') as exp_file:
        fnames = [' ',' ','PSO',' ',' ','GA',' ']
        writer = csv.DictWriter(exp_file, fieldnames=fnames)
        writer.writeheader()

        fnames = ['File','Error PSO','Running time PSO','Generations PSO','Error GA','Running time GA','Generations GA']
        writer = csv.DictWriter(exp_file, fieldnames=fnames)
        writer.writeheader()

        for i in range(len(r1)):
            assert(r1[i]['file'] == r2[i]['file'])
            writer.writerow({
                'File' : r1[i]['file'],
                'Error PSO' : r1[i]['error'],
                'Running time PSO' : r1[i]['running time'],
                'Generations PSO' : r1[i]['generations'],
                'Error GA' : r2[i]['error'],
                'Running time GA' : r2[i]['running time'],
                'Generations GA' : r2[i]['generations']
            })

        # fnames = ['Mean PSO','Variance PSO','Running time PSO','Generations PSO','Mean GA','Variance GA','Running time GA','Generations GA']
        # writer = csv.DictWriter(exp_file, fieldnames=fnames)
        # writer.writeheader()
        # writer.writerow({
            # 'Mean PSO' : mean([x['error'] for x in r1]),
            # 'Variance PSO' : variance([x['error'] for x in r1]),
            # 'Running time PSO': mean([x['running time'] for x in r1]),
            # 'Generations PSO' : mean([x['generations'] for x in r1]),
            # 'Mean GA' : mean([x['error'] for x in r2]),
            # 'Variance GA' : variance([x['error'] for x in r2]),
            # 'Running time GA': mean([x['running time'] for x in r2]),
            # 'Generations GA' : mean([x['generations'] for x in r2])
        # })

model = [spso,sga]
times = 30

def summarize_each_record(record):
    avg_err = sum(x['error'] for x in record) / len(record)
    best_err = min(x['error'] for x in record)
    worst_err = max(x['error'] for x in record)

    avg_rt = sum(x['running time'] for x in record) / len(record)
    best_rt = min(x['running time'] for x in record)
    worst_rt = max(x['running time'] for x in record)

    return {
        'avg_err'  : avg_err,
        'best_err' : best_err,
        'worst_err': worst_err,
        'avg_rt'   : avg_rt,
        'best_rt'  : best_rt,
        'worst_rt' : worst_rt
    }

def summarize(records):
    ret = defaultdict(lambda: {})

    for key in records.keys():
        value = summarize_each_record(records[key])

        ret[key] = value

    return ret

def write_to_file(filename, result):
    f = open(filename,"w")
    
    f.write("dataset\t\t\tavg_err\t\tbest_err\tworst_err\tavg_rt\t\tbest_rt\t\tworst_rt\n")

    for key in result.keys():
        res = result[key]

        f.write("{:20}\t\t{:8.3f}\t\t{:8.3}\t{:8.3f}\t{:8.3f}\t\t{:8.3f}\t\t{:8.3f}\n".format(
            key,res['avg_err'],res['best_err'],res['worst_err'],
            res['avg_rt'],res['best_rt'],res['worst_rt']
        ))

def run(filesize, fn, size):
    ga_dataset = defaultdict(lambda: [])
    pso_dataset = defaultdict(lambda: [])

    for time in range(times):
        r = []
        for m in model:
            r.append(cal(filesize,m,run_with_same_generations))

        r1,r2 = r
        r1.sort(key=lambda x: x['file'])
        r2.sort(key=lambda x: x['file'])

        for record in r1:
            ga_dataset[record['file']].append(record)

        for record in r2:
            pso_dataset[record['file']].append(record)
    
    if size == "small":
        write_to_file("SmallGA", summarize(ga_dataset))
        write_to_file("SmallPSO", summarize(pso_dataset))
    else:
        write_to_file("MediumGA", summarize(ga_dataset))
        write_to_file("MediumSO", summarize(pso_dataset))

    # export_csv(fn,r1,r2)

run(small,'small.csv', "small")
run(medium,'medium.csv', "medium")
