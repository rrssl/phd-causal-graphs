"""
Processing domino-pair samples. For each sample, test whether if the first
domino is just out of equilibrium, the second topples.

Parameters
----------
spath : string
  Path to the samples to process.
sid : int
  ID of the scenario used to generate the samples. See sampling_method.py.

"""
import os
import sys

import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from config import NCORES
sys.path.insert(0, os.path.abspath("../.."))
import xp.simulate as simu  # noqa
from xp.sampling_methods import Scenario, sample2coords  # noqa


def run_domino_toppling_xp(coords, visual=False):
    """
    Run the domino toppling simulation. If not visual, returns True if
    the second domino topples.

    Parameters
    ----------
    coords : array
        Sequence of (x, y, angle) global coordinates of each domino.
    visual : boolean
        Run the experiment in 'visual' mode, that is, actually display the
        scene in a window. In that case, 'timestep' and 'maxtime' are ignored.
    """
    doms_np, world = simu.setup_dominoes(coords, _make_geom=visual)

    if visual:
        simu.run_simu(doms_np, world, _visual=True)
        return True
    else:
        doms = list(doms_np.get_children())
        for d1, d2 in zip(doms[:-1], doms[1:]):
            test = world.contact_test_pair(d1.node(), d2.node())
            if test.get_num_contacts() > 0:
                return False

        times = simu.run_simu(doms_np, world)
        return np.isfinite(times).all()


def _test_domino_toppling_xp():
    assert run_domino_toppling_xp([[0, 0, 0], [.02, .01, 15]], 0)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])
    samples = np.load(spath)
    coords_list = sample2coords(samples, Scenario(sid))
    values = Parallel(n_jobs=NCORES)(
            delayed(run_domino_toppling_xp)(coords)
            for coords in coords_list)
    #  print(values)
    root = os.path.splitext(spath)[0]
    np.save(root + "-labels.npy", values)


if __name__ == "__main__":
    main()
