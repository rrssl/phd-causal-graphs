"""
Process all candidate splines with one of the methods.

Parameters
----------
mid : int
  Method id. See methods.py for the list of methods.
spath : string
  Path to the .pkl list of splines
ns : int, optional
  Only process the ns first splines of the list.

"""
import os
import sys
import pickle
import time

import numpy as np
from scipy.interpolate import splev
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel

from config import NCORES
from methods import get_methods


# We use a callable object here instead of a decorator so that it can be
# pickled and thus, calls can be parallelized. (This is a limitation of
# Pickle).
class TimedMethod:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        t1 = time.time()
        output = self.f(*args)
        t2 = time.time()
        return output, t2 - t1


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    mid = int(sys.argv[1])
    spath = sys.argv[2]
    ns = int(sys.argv[3]) if len(sys.argv) == 4 else None

    with open(spath, 'rb') as fin:
        splines = pickle.load(fin)[slice(ns)]

    method = get_methods()[mid-1]
    timed_method = TimedMethod(method)
    out = Parallel(n_jobs=NCORES)(
            delayed(timed_method)(spline) for spline in splines)
    #  out = [timed_method(spline) for spline in splines]
    results, times = zip(*out)  # Unzip!

    dirname = os.path.dirname(spath)
    prefix = os.path.splitext(os.path.basename(spath))[0]
    outname = prefix + "-dominoes-method_{}.npz".format(mid)
    np.savez(os.path.join(dirname, outname), *results)
    outname = prefix + "-times-method_{}.npy".format(mid)
    np.save(os.path.join(dirname, outname), times)


if __name__ == "__main__":
    main()
