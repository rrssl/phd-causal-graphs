"""
Processing scenario instances.

Parameters
----------
spath : string
  Path to the samples to process.
sid : int
  ID of the scenario used to generate the samples. See gen_training_set.py.
n : int, optional
  Additional scenario parameter, depends on the scenario.

"""
import os
import sys

import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from config import NCORES
sys.path.insert(0, os.path.abspath("../.."))
from xp.simulate import Simulation  # noqa: E402
from xp.scenarios import SCENARIOS  # noqa: E402


def process(scenario, sample, n=0, visual=False):
    """
    Run the simulation. If not visual, returns True if the second domino
    topples.

    Parameters
    ----------
    scenario : object
      Class of the chosen scenario (from scenarios.py).
    sample : array
      Parameters for this instance of the scenario.
    n : int, optional
      Additional parameter for the scenario.
    visual : bool, optional
      Run the experiment in 'visual' mode, that is, actually display the
      scene in a window.

    """
    instance = scenario(sample, n, make_geom=visual)
    simu = Simulation(instance)

    if visual:
        simu.run_visual()
        return True

    simu.run()
    return instance.succeeded()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])
    try:
        n = int(sys.argv[3])
    except IndexError:
        n = 0
    scenario = SCENARIOS[sid]
    samples = np.load(spath)
    values = Parallel(n_jobs=NCORES)(
            delayed(process)(scenario, sample, n) for sample in samples)
    #  print(values)
    root = os.path.splitext(spath)[0]
    np.save(root + "-labels.npy", values)


if __name__ == "__main__":
    main()
