"""
Process all candidate splines with one of the methods.

Parameters
----------
mid : int
  Method id. See methods.py for the list of methods.
spath : string
  Path to the .pkl list of splines

"""
import os
import sys
import pickle
import time

import numpy as np
from scipy.interpolate import splev

from methods import get_methods


def time_calls(f, times):
    def wrap(*args, **kwargs):
        t1 = time.time()
        output = f(*args, **kwargs)
        t2 = time.time()
        times.append(t2 - t1)
        return output
    return wrap


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    mid = int(sys.argv[1])
    spath = sys.argv[2]

    with open(spath, 'rb') as fin:
        splines = pickle.load(fin)

    method = get_methods()[mid-1]
    times = []
    timed_method = time_calls(method, times)
    results = [timed_method(spline) for spline in splines]

    dirname = os.path.dirname(spath)
    outname = "dominoes-method_{}.npz".format(mid)
    np.savez(os.path.join(dirname, outname), *results)
    outname = "times-method_{}.npy".format(mid)
    np.save(os.path.join(dirname, outname), times)


if __name__ == "__main__":
    main()
