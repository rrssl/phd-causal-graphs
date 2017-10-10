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

from config import t, w, h, MASS
from config import TIMESTEP, MAX_WAIT_TIME
from config import NCORES
from functions import run_domino_toppling_xp


def process(samples):
    if samples.shape[1] == 2:
        values = Parallel(n_jobs=NCORES)(
                delayed(run_domino_toppling_xp)(
                    (t, w, h, d, 0, a, MASS), TIMESTEP, MAX_WAIT_TIME)
                for d, a in samples)
    else:
        values = Parallel(n_jobs=NCORES)(
                delayed(run_domino_toppling_xp)(
                    (t, w, h, x, y, a, MASS), TIMESTEP, MAX_WAIT_TIME)
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
