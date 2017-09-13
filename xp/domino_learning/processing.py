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

from functions import run_domino_toppling_xp
from config import timestep, maxtime, density, t, w, h


def process(samples):
    m = density * t * w * h
    values = np.empty(len(samples))
    if samples.shape[1] == 2:
        for i, (d, a) in enumerate(samples):
            values[i] = run_domino_toppling_xp(
                    (t, w, h, d, 0, a, m), timestep, maxtime)
    else:
        for i, (x, y, a) in enumerate(samples):
            values[i] = run_domino_toppling_xp(
                    (t, w, h, x, y, a, m), timestep, maxtime)

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
