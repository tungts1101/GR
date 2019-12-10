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
            'running time' : r['running time'],
            'generations' : r['generations']
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

model = [spso,sga]
times = 30

def summarize_each_record(record):
    f = record[0]['file']
    avg_err = sum(x['error'] for x in record) / len(record)
    best_err = min(x['error'] for x in record)
    worst_err = max(x['error'] for x in record)

    avg_rt = sum(x['running time'] for x in record) / len(record)
    best_rt = min(x['running time'] for x in record)
    worst_rt = max(x['running time'] for x in record)
    
    avg_gen = sum(x['generations'] for x in record) // len(record)
    best_gen = min(x['generations'] for x in record)
    worst_gen = max(x['generations'] for x in record)

    return {
        'file'     : f,
        'avg_err'  : avg_err,
        'best_err' : best_err,
        'worst_err': worst_err,
        'avg_rt'   : avg_rt,
        'best_rt'  : best_rt,
        'worst_rt' : worst_rt,
        'avg_gen'  : avg_gen,
        'best_gen' : best_gen,
        'worst_gen': worst_gen
    }

def summarize(records):
    ret = defaultdict(lambda: {})

    for key in records.keys():
        value = summarize_each_record(records[key])

        ret[key] = value

    return ret

# def write_to_file(filename, result):
    # f = open(filename,"w")
    
    # f.write("dataset\t\t\tavg_err\t\tbest_err\tworst_err\tavg_rt\t\tbest_rt\t\tworst_rt\n")

    # for key in result.keys():
        # res = result[key]

        # f.write("{:20}\t\t{:8.3f}\t\t{:8.3}\t{:8.3f}\t{:8.3f}\t\t{:8.3f}\t\t{:8.3f}\t\t{}\n".format(
            # key,res['avg_err'],res['best_err'],res['worst_err'],
            # res['avg_rt'],res['best_rt'],res['worst_rt']
        # ))

def export_csv(csvf,result):
    with open(csvf,mode='w') as exp_file:
        fnames = ['File','Avg Err','Best Err','Worst Err','Avg Rt','Best Rt','Worst Rt','Avg Gen','Best Gen','Worst Gen']

        writer = csv.DictWriter(exp_file, fieldnames=fnames)
        writer.writeheader()

        for key in result.keys():
            res = result[key]

            writer.writerow({
                'File'     : res['file'],
                'Avg Err'  : res['avg_err'],
                'Best Err' : res['best_err'],
                'Worst Err': res['worst_err'],
                'Avg Rt'   : res['avg_rt'],
                'Best Rt'  : res['best_rt'],
                'Worst Rt' : res['worst_rt'],
                'Avg Gen'  : res['avg_gen'],
                'Best Gen' : res['best_gen'],
                'Worst Gen': res['worst_gen']
            })


def eval_sg(filesize, size):
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
        export_csv("SG_Small_GA.csv", summarize(ga_dataset))
        export_csv("SG_Small_PSO.csv", summarize(pso_dataset))
    else:
        export_csv("SG_Medium_GA.csv", summarize(ga_dataset))
        export_csv("SG_Medium_PSO.csv", summarize(pso_dataset))

eval_sg(small, "small")
eval_sg(medium, "medium")

def eval_dg(filesize, size):
    ga_dataset = defaultdict(lambda: [])
    pso_dataset = defaultdict(lambda: [])

    for time in range(times):
        r = []
        for m in model:
            r.append(cal(filesize,m,run_with_diff_generations))

        r1,r2 = r
        r1.sort(key=lambda x: x['file'])
        r2.sort(key=lambda x: x['file'])

        for record in r1:
            ga_dataset[record['file']].append(record)

        for record in r2:
            pso_dataset[record['file']].append(record)
    
    if size == "small":
        export_csv("DG_Small_GA.csv", summarize(ga_dataset))
        export_csv("DG_Small_PSO.csv", summarize(pso_dataset))
    else:
        export_csv("DG_Medium_GA.csv", summarize(ga_dataset))
        export_csv("DG_Medium_PSO.csv", summarize(pso_dataset))

eval_dg(small, "small")
eval_dg(medium,"medium")
