"""
Evaluate different ways of estimating the toppling time of domino chains.

Parameters
----------
mid : int
  Index of the evaluation method. See methods.py for details. Note that mid==0
  should be called before any other, as it will produce a reference for
  failure points.

"""
import os
import pickle
import sys
import timeit

import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from config import NCORES
sys.path.insert(0, os.path.abspath(".."))
from predicting_domino_timing.methods import get_methods


SPL_PATH = "../domino_design/data/latest/candidates.pkl"
DOM_PATH = "../domino_design/data/latest/candidates-dominoes-method_2.npz"
OUT_DIR = "data/latest/"


def eval_method(method, distrib, spline):
    t = timeit.default_timer()
    top_times = method(distrib, spline)
    comp_time = timeit.default_timer() - t
    ind = np.argmax(top_times)
    # Check for chain failure (i.e. infinite time)
    if top_times[ind] == np.inf:
        ind -= 1
    phys_time = top_times[ind]  # Total toppling time, until failure or end
    return phys_time, comp_time, ind


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    mid = int(sys.argv[1])

    with open(SPL_PATH, 'rb') as f:
        splines = pickle.load(f)
    doms = np.load(DOM_PATH)
    method = get_methods()[mid]

    if mid == 0:
        results = Parallel(n_jobs=NCORES)(
                delayed(eval_method)(
                    method, doms['arr_{}'.format(i)], spline)
                for i, spline in enumerate(splines))
        phys_times, comp_times, inds = zip(*results)
        # For this method we also record the last toppling domino index.
        np.save(OUT_DIR + "last-top-inds.npy", inds)
    else:
        inds = np.load(OUT_DIR + "last-top-inds.npy")
        results = Parallel(n_jobs=NCORES)(
                delayed(eval_method)(
                    method, doms['arr_{}'.format(i)][:ind+1], spline)
                for i, (ind, spline) in enumerate(zip(inds, splines)))
        phys_times, comp_times, _ = zip(*results)
    method_name = "method_{}".format(mid)
    np.save(OUT_DIR + method_name + "-phys-times.npy", phys_times)
    np.save(OUT_DIR + method_name + "-comp-times.npy", comp_times)


if __name__ == "__main__":
    main()
