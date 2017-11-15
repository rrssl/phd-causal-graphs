"""
Compute a domino distribution for the input BSpline path.

Parameters
----------
spath : string
  Path to the .pkl file of splines.
sid : int
  Index of the spline in the file.

"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from xp.domino_design.methods import batch_classif_based
from xp.viewdoms import show_dominoes


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])

    with open(spath, 'rb') as f:
        spline = pickle.load(f)[sid]

    u = batch_classif_based(spline, batchsize=2)
    show_dominoes([u], [spline])

    filename = os.path.splitext(spath)[0] + "-doms.npz"
    np.savez(filename, u)


if __name__ == "__main__":
    main()
