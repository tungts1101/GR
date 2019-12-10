import pulp
import os
import joblib
import importlib
import logging
import logzero
import numpy as np


from logzero import logger
from argparse import ArgumentParser
from pulp import LpVariable

import common.point as point

from common.input import WusnInput, WusnConstants
from common.output import WusnOutput


def model_lp(inp: WusnInput, alpha=0.5, lax=False):
    var_cls = pulp.LpBinary if not lax else 'Continous'
    sensor_loss = inp.sensor_loss
    sensors, relays = list(inp.sensors), list(inp.relays)
    n, m = inp.num_of_sensors, inp.num_of_relays
    E_Rx, E_Da = WusnConstants.e_rx, WusnConstants.e_da
    SL = _get_sn_loss_matrix(sensor_loss, sensors, relays)
    e_mp = WusnConstants.e_mp
    C = _get_conn_matrix(sensor_loss, sensors, relays)
    dB = _get_bs_distances(relays, inp.BS)
    Emax = inp.e_max

    prob = pulp.LpProblem('RelaySelection', pulp.LpMinimize)

    # Variables
    # Z = [LpVariable('z_%d' % j, lowBound=0, upBound=1, cat=var_cls) for j in range(m)]
    Z = [LpVariable('z_%d' % j, lowBound=0, upBound=1, cat=pulp.LpBinary) for j in range(m)]
    Z = np.asarray(Z, dtype=object)
    A = []
    for i in range(n):
        row = [LpVariable('a_%d_%d' % (i, j), lowBound=0, upBound=1, cat=var_cls) for j in range(m)]
        A.append(row)
    A = np.asarray(A, dtype=object)
    Ex = LpVariable('Ex', lowBound=0)
    Er = [LpVariable('Er_%d' % j) for j in range(m)]
    Er = np.asarray(Er, dtype=object)

    # Constraints
    for j in range(m):
        asum = pulp.lpSum(A[:, j])
        prob += (Er[j] == (WusnConstants.k_bit * (asum * (E_Rx + E_Da)) +
                           WusnConstants.k_bit * Z[j] * e_mp * (dB[j] ** 4)))
        prob += ((asum - Z[j] * n) <= 0)
        prob += (asum >= Z[j])
        prob += (Ex >= Er[j])
    for i in range(n):
        prob += (pulp.lpSum(A[i]) == 1)
    for i in range(n):
        for j in range(m):
            prob += (A[i, j] <= C[i, j])
            prob += (Ex >= A[i, j] * SL[i, j])

    Cx = alpha / m * pulp.lpSum(Z) + (1 - alpha) / Emax * Ex
    prob.setObjective(Cx)

    return prob


def output_from_prob(prob: pulp.LpProblem, inp: WusnInput):
    out = WusnOutput(inp)
    vars_ = prob.variablesDict()
    for i, sn in enumerate(inp.sensors):
        for j, rn in enumerate(inp.relays):
            assgn = vars_['a_%d_%d' % (i, j)].value()
            if assgn > 0:
                out.assign(rn, sn)
                break
    return out


def solve(inp: WusnInput, save_path, alpha=0.5, lax=False):
    lz = importlib.reload(logzero)

    def log(msg, level=logging.INFO):
        lz.logger.log(level, '[%s] %s' % (save_path, msg))
    if os.path.exists(save_path):
        log('Exist')
        return
    inp.freeze()
    try:
        log('Modeling LP')
        prob = model_lp(inp, alpha, lax)
        log('Solving LP')
        prob.solve()
    
        if prob.status == pulp.LpStatusOptimal:
            log('Converting')
            out = output_from_prob(prob, inp)
            log('Saving')
            # out.to_file(save_path)
            # with open('')
            with open(save_path, 'w+') as f:
                f.write('[%s] %.8e' % (save_path, prob.objective.value()))
        else:
            log('Unsolvable', level=logging.WARN)
            print('[%s] UNSOLVED' % (save_path,))

    except KeyboardInterrupt:
        log('Canceled')
        return
    except Exception as e:
        raise e


def _get_sn_loss_matrix(sensor_loss, sensors, relays):
    n, m = len(sensors), len(relays)
    mat = np.zeros((n, m), dtype=np.float)
    for i, sn in enumerate(sensors):
        for j, rn in enumerate(relays):
            if (sn, rn) in sensor_loss.keys():
                mat[i, j] = sensor_loss[(sn, rn)]
    return mat


def _get_bs_distances(relays, base_station):
    ds = []
    for rn in relays:
        ds.append(point.distance(rn, base_station))
    return ds


def _get_conn_matrix(sensor_loss, sensors, relays,):
    n, m = len(sensors), len(relays)
    mat = np.zeros((n, m), dtype=np.int)
    for i, sn in enumerate(sensors):
        for j, rn in enumerate(relays):
            if (sn, rn) in sensor_loss.keys():
                mat[i, j] = 1
    return mat


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='input', nargs='+', help='Input files. Accept globs as input.')
    parser.add_argument('-p', '--procs', type=int, default=4, help='Number of processes to fork')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha coefficient')
    parser.add_argument('--lax', action='store_true')
    parser.add_argument('-o', '--outdir', default='results/exact')
    parser.add_argument('-i', '--iteration', default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_arguments()
    args_.input = [x for x  in args_.input if 'r25' in x and 'uu' not in x]

    os.makedirs(args_.outdir, exist_ok=True)
    logger.info('Loading input files')
    save_paths = args_.input
    save_paths = map(lambda x: os.path.split(x)[-1].split('.')[0], save_paths)
    save_paths = map(lambda x: os.path.join(args_.outdir, x), save_paths)
    save_paths = map(lambda x: x + '.out' + str(args_.iteration), save_paths)
    save_paths = [x for x in list(save_paths)]

    inputs = [WusnInput.from_file(x) for x in args_.input]
    for i,j in zip([x for x in args_.input], save_paths):
        if i.split('/')[-1].split('.')[0] in j == False:
            print(i,j)
    
    print(len(save_paths), len(inputs))

    logger.info('Solving %d problems' % len(inputs))
    if args_.lax:
        logger.info('Approximating...')

    logger.info('Running using %d workers' % args_.procs)
    joblib.Parallel(n_jobs=args_.procs)(
        joblib.delayed(solve)(inp, sp, args_.alpha, args_.lax) for inp, sp in zip(inputs, save_paths)
    )
