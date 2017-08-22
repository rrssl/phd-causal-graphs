"""
Process all candidate paths with one of the methods.

Parameters
----------
mid : int
  Method id. See below.
pickle_path : string
  Path to the .pkl file of curves

Methods:
  1. Equal spacing
  2. Minimal spacing
  3. Incremental classifier-based spacing

"""
import os
import sys
import time
import pickle

import numpy as np
from scipy.interpolate import splev

from methods import equal_spacing, minimal_spacing, inc_classif_based


def timef(f, times):
    def wrap(*args, **kwargs):
        t1 = time.time()
        output = f(*args, **kwargs)
        t2 = time.time()
        times.append(t2 - t1)
        return output
    return wrap


def main():
    if len(sys.argv) < 3:
        print("Please provide a method number and the location of the paths.")
        return
    mid = int(sys.argv[1])
    pickle_path = sys.argv[2]

    with open(pickle_path, 'rb') as fin:
        domino_paths = pickle.load(fin)

    method = (equal_spacing, minimal_spacing, inc_classif_based)[mid]
    #  times = []
    #  timed_method = timef(method, times)
    results = [method(path) for path in domino_paths]

    with open("dominoes-method_{}.pkl".format(mid), 'wb') as fout:
        pickle.dump(results, fout)
    #  with open("times-method_{}.pkl".format(mid), 'wb') as ftimes:
        #  pickle.dump(times, ftimes)


if __name__ == "__main__":
    main()
