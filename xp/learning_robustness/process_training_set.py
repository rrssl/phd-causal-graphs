"""
Processing scenario instances.

Parameters
----------
spath : string
  Path to the samples to process.
sid : int
  ID of the scenario used to generate the samples. See gen_training_set.py.
argname : string, optional
  Name of the additional scenario parameter.
argval : int, optional
  Value associated to the additional scenario parameter.

"""
import os
import sys

import numpy as np
from sklearn.externals.joblib import Parallel, delayed

from config import NCORES
sys.path.insert(0, os.path.abspath("../.."))
from xp.simulate import Simulation  # noqa: E402
from xp.scenarios import SCENARIOS  # noqa: E402

VISUAL = 0


def process(scenario, sample, visual=False, **kwargs):
    """
    Run the simulation. If not visual, returns True if the second domino
    topples.

    Parameters
    ----------
    scenario : object
      Class of the chosen scenario (from scenarios.py).
    sample : array
      Parameters for this instance of the scenario.
    visual : bool, optional
      Run the experiment in 'visual' mode, that is, actually display the
      scene in a window.
    kwargs : dict, optional
      Key-value pair of additional parameters for the scenario.

    """
    instance = scenario(sample, make_geom=visual, **kwargs)
    simu = Simulation(instance)

    if visual:
        simu.run_visual()
    else:
        simu.run()

    return instance.succeeded()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])
    argit = iter(sys.argv[3:])
    kwargs = {argname: int(argval) for argname, argval in zip(argit, argit)}
    print("kwargs: ", kwargs)
    scenario = SCENARIOS[sid]
    samples = np.load(spath)
    values = Parallel(n_jobs=1 if VISUAL else NCORES)(
            delayed(process)(scenario, sample, visual=VISUAL, **kwargs)
            for sample in samples)
    print(values)
    root = os.path.splitext(spath)[0]
    np.save(root + "-labels.npy", values)


if __name__ == "__main__":
    main()
