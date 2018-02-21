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
from sklearn.externals.joblib import delayed, Parallel

from config import NCORES
sys.path.insert(0, os.path.abspath("../.."))
import xp.simulate as simu


VERBOSE = 0


def compute_times(u, spline, _id=None):
    if VERBOSE:
        print("Simulating distribution {}".format(_id))
    doms_np, world = simu.setup_dominoes_from_path(u, spline)
    toppling_times = simu.run_simu(doms_np, world)
    if VERBOSE:
        print("Done with distribution {}".format(_id))
    return toppling_times


def main():
    if len(sys.argv) < 2:
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
