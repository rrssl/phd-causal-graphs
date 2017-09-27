"""
Processing domino-pair samples. For each sample, test whether if the first
domino is just out of equilibrium, the second topples.

Parameters
----------
spath : string
  Path to the samples to process.

"""
import os
import sys

import numpy as np
from sklearn.externals.joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath('..'))
from predicting_domino_toppling.functions import run_domino_toppling_xp
from predicting_domino_toppling.config import density, t, w, h
from predicting_domino_toppling.config import timestep, MAX_WAIT_TIME
from predicting_domino_toppling.config import NCORES


def process(samples):
    m = density * t * w * h
    if samples.shape[1] == 2:
        values = Parallel(n_jobs=NCORES)(
                delayed(run_domino_toppling_xp)(
                    (t, w, h, d, 0, a, m), timestep, MAX_WAIT_TIME)
                for d, a in samples)
    else:
        values = Parallel(n_jobs=NCORES)(
                delayed(run_domino_toppling_xp)(
                    (t, w, h, x, y, a, m), timestep, MAX_WAIT_TIME)
                for x, y, a in samples)

    return values


def main():
    if len(sys.argv) <= 1:
        print(__doc__)
        return
    spath = sys.argv[1]
    samples = np.load(spath)
    assert samples.shape[1] in (2, 3), "Number of dimensions must be 2 or 3."
    values = process(samples)
    #  print(values)
    root = os.path.splitext(spath)[0]
    np.save(root + "-values.npy", values)


if __name__ == "__main__":
    main()
