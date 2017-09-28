"""
Generate the domino distributions from the candidate splines.

Parameters
----------
spath : string
  Path to the .pkl file of candidate splines.
nvar : int
  Number of variations to generate per spline.

"""
import math
import os
import pickle
import random
import sys

import numpy as np
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel

from config import t, h
from config import NCORES, MIN_DENSITY, MAX_DENSITY
sys.path.insert(0, os.path.abspath(".."))
from domino_design.methods import equal_spacing
from domino_design.evaluate import test_no_successive_overlap_fast
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


VERBOSE = False


def distribute_dominoes(spline, density):
    length = spl.arclength(spline)
    ndoms = math.floor(length * density / t)
    u = equal_spacing(spline, ndoms)
    return u


def generate_distributions(spline, nvar=1, _id=None):
    length = spl.arclength(spline)
    distribs = []
    for i in range(nvar):
        if VERBOSE:
            print("Processing spline {}: variation {}".format(_id, i))
        while True:
            density = random.uniform(MIN_DENSITY, MAX_DENSITY)
            u = distribute_dominoes(spline, density)
            if test_no_successive_overlap_fast(u, spline):
                distribs.append((density, np.array(u)))
                break
    return distribs


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    nvar = int(sys.argv[2])

    with open(spath, 'rb') as f:
        splines = pickle.load(f)

    distrib_lists = Parallel(n_jobs=NCORES)(
            delayed(generate_distributions)(spline, nvar, i)
            for i, spline in enumerate(splines))
    # Flatten the list
    distribs = [distrib for distrib_list in distrib_lists
                for distrib in distrib_list]
    densities, dominoes = zip(*distribs)
    dom2spl = [i for i in range(len(splines)) for _ in range(nvar)]

    root, _ = os.path.splitext(spath)
    np.savez(root + "-doms.npz", *dominoes)
    np.save(root + "-densities.npy", np.array(densities))
    np.save(root + "-dom2spl.npy", np.array(dom2spl))


if __name__ == "__main__":
    main()
