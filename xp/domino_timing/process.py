"""
Run the simulation for each domino distribution and record the toppling time
of each domino.

Parameters
----------
spath : string
  Path to the .pkl file of candidate splines.

"""
import os
import pickle
import sys

import numpy as np
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel

from config import NCORES, MAX_WAIT_TIME
sys.path.insert(0, os.path.abspath(".."))
from domino_design.evaluate import setup_dominoes, get_toppling_angle


VERBOSE = True


def compute_times(u, spline, _id):
    if VERBOSE:
        print("Simulating distribution {}".format(_id))
    doms_np, world = setup_dominoes(u, spline)
    n = len(u)
    dominoes = list(doms_np.get_children())
    last_toppled_id = -1
    toppling_times = np.full(n, np.inf)
    time = 0.
    toppling_angle = get_toppling_angle()
    while True:
        if dominoes[last_toppled_id+1].get_r() >= toppling_angle:
            last_toppled_id += 1
            toppling_times[last_toppled_id] = time
        if last_toppled_id == n-1:
            # All dominoes toppled in order.
            break
        if time - toppling_times[last_toppled_id] > MAX_WAIT_TIME:
            # The chain broke
            break
        time += 1/120
        world.do_physics(1/120, 2, 1/120)
    if VERBOSE:
        print("Done with distribution {}".format(_id))
    return toppling_times


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    spath = sys.argv[1]
    root, _ = os.path.splitext(spath)
    dpath = root + "-doms.npz"
    d2spath = root + "-dom2spl.npy"

    with open(spath, 'rb') as f:
        splines = pickle.load(f)
    doms = np.load(dpath)
    dom2spl = np.load(d2spath)

    times = Parallel(n_jobs=NCORES)(
            delayed(compute_times)(doms['arr_{}'.format(i)],
                                   splines[dom2spl[i]], i)
            for i in range(len(doms.files)))

    np.savez(root + "-times.npz", *times)


if __name__ == "__main__":
    main()
