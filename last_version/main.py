import os

fp = os.path.join(os.path.dirname(__file__),'../WusnNewModel/data')

def get_dataset(d):
    files = []
    for f in os.listdir(d):
        if f[-2:] == 'in':
            files.append(d+f)
    return files

small = get_dataset(fp+'/small_data')
medium = get_dataset(fp+'/medium_data')

def handle_dataset(f,swarm,res):
    nw = Network(f)
    s = swarm(nw)
    r = s.eval()

    res.append(
        {
            'file': os.path.split(os.path.abspath(f))[1],
            'error': r['error'],
            'running time': r['running time'],
            'generations': r['generations']
        }
    )

def run(fl,swarm):
    res = Manager().list()


if __name__ == '__main__':

