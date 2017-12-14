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

from config import NCORES
sys.path.insert(0, os.path.abspath("../.."))
import xp.simulate as simu


def run_domino_toppling_xp(params, visual=False):
    """
    Run the domino-pair toppling simulation. If not visual, returns True if
    the second domino topples.

    Parameters
    ----------
    params : sequence
        Parameter vector (x, y, angle), i.e. D2's coordinates relative to D1.
    visual : boolean
        Run the experiment in 'visual' mode, that is, actually display the
        scene in a window. In that case, 'timestep' and 'maxtime' are ignored.
    """
    x, y, a = params
    global_coords = [[0, 0, 0], [x, y, a]]
    doms_np, world = simu.setup_dominoes(global_coords, _make_geom=visual)

    if visual:
        simu.run_simu(doms_np, world, _visual=True)
        return True
    else:
        d1, d2 = doms_np.get_children()
        test = world.contact_test_pair(d1.node(), d2.node())
        if test.get_num_contacts() > 0:
            return False

        times = simu.run_simu(doms_np, world)
        return np.isfinite(times).all()


def _test_domino_toppling_xp():
    assert run_domino_toppling_xp((.02, .01, 15), 0)


def process(samples):
    if samples.shape[1] == 2:
        samples_ = np.empty((samples.shape[0], 3))
        samples_[:, 0] = samples[:, 0] * np.cos(samples[:, 1] * np.pi / 180)
        samples_[:, 1] = samples[:, 0] * np.sin(samples[:, 1] * np.pi / 180)
        samples_[:, 2] = samples[:, 1]
        values = Parallel(n_jobs=NCORES)(
                delayed(run_domino_toppling_xp)((x, y, a))
                for x, y, a in samples_)
    else:
        values = Parallel(n_jobs=NCORES)(
                delayed(run_domino_toppling_xp)((x, y, a))
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
    np.save(root + "-labels.npy", values)


if __name__ == "__main__":
    main()
