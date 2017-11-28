"""
Run the simulation for each training sample and record the topple-to-topple
time.

Parameters
----------
spath : string
  Path to the .npy samples.
nprev : int
  Number of dominoes to place before the pair of interest.

"""
import os
import sys

import numpy as np
from sklearn.externals.joblib import delayed, Parallel

from config import NCORES
sys.path.insert(0, os.path.abspath("../.."))
import xp.simulate as simu


def compute_toppling_time(x, y, a, s, nprev, _visual=False):
    length = s * nprev
    # Typical global coordinates (s=.5, nprev=2):
    # [[ -1, 0, 0],
    #  [-.5, 0, 0],
    #  [  0, 0, 0],
    #  [  x, y, a]]
    global_coords = np.zeros((nprev+2, 3))
    global_coords[:-1, 0] = np.linspace(-length, 0, nprev+1)
    global_coords[-1] = x, y, a
    doms_np, world = simu.setup_dominoes(global_coords, _make_geom=_visual)

    if _visual:
        simu.run_simu(doms_np, world, _visual=True)
        return

    toppling_times = simu.run_simu(doms_np, world)
    if np.isfinite(toppling_times).all():
        return toppling_times[-1] - toppling_times[-2]
    else:
        return np.inf


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    spath = sys.argv[1]
    nprev = int(sys.argv[2])

    samples = np.load(spath)
    assert samples.shape[1] == 4, "Number of dimensions must be 4."
    times = Parallel(n_jobs=NCORES)(
            delayed(compute_toppling_time)(x, y, a, s, nprev)
            for x, y, a, s in samples)

    root, _ = os.path.splitext(spath)
    np.save(root + "-times-{}.npy".format(nprev), times)


if __name__ == "__main__":
    main()
