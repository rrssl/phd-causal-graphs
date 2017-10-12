"""
Run the simulation for each domino pair and record the toppling-to-toppling
time.

Parameters
----------
spath : string
  Path to the .npy samples.

"""
import os
import sys

import numpy as np
from sklearn.externals.joblib import delayed, Parallel

sys.path.insert(0, os.path.abspath("../.."))
from xp.config import NCORES
import xp.simulate as simu


def run_predicting_domino_timing_xp(params):
    x, y, a = params
    global_coords = [[0, 0, 0], [x, y, a]]
    doms_np, world = simu.setup_dominoes(global_coords)

    d1, d2 = doms_np.get_children()
    test = world.contact_test_pair(d1.node(), d2.node())
    if test.get_num_contacts() > 0:
        return np.inf

    toppling_times = simu.run_simu(doms_np, world)
    return toppling_times.max()


def compute_times(samples):
    if samples.shape[1] == 2:
        times = Parallel(n_jobs=NCORES)(
                delayed(run_predicting_domino_timing_xp)(d, 0, a)
                for d, a in samples)
    else:
        times = Parallel(n_jobs=NCORES)(
                delayed(run_predicting_domino_timing_xp)(x, y, a)
                for x, y, a in samples)
    return times


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    spath = sys.argv[1]
    samples = np.load(spath)
    assert samples.shape[1] in (2, 3), "Number of dimensions must be 2 or 3."
    times = compute_times(samples)
    root, _ = os.path.splitext(spath)
    np.save(root + "-times.npy", times)


if __name__ == "__main__":
    main()
