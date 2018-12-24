"""
Compute the success rate decay.

Parameters
----------
path : string
  Path to the scenario file.
n_samples : int
  Number of samples in the dataset.
T : int
  Duration of the simulation in seconds.

"""
import os
import sys

import numpy as np
from joblib import delayed, Memory, Parallel
from timeit import default_timer as timer

sys.path.insert(0, os.path.abspath(".."))
import core.robustness as rob  # noqa: E402
from core.scenario import (import_scenario_data, load_scenario)  # noqa: E402

FPS = 500
NCORES = 2
memory = Memory(cachedir=".cache")


# Parallelized
def compute_label(scenario, x, **simu_kw):
    if scenario.check_physically_valid_sample(x):
        label = 2*int(rob.compute_label(scenario, x, **simu_kw)) - 1
    else:
        label = 0
    return label


# Memoized
@memory.cache
def generate_dataset(scenario, n_samples, **simu_kw):
    t = timer()
    ndims = len(scenario.design_space)
    X = rob.MultivariateUniform(ndims).sample(n_samples)
    # labels = [compute_label(scenario, x, **simu_kw) for x in X]
    y = Parallel(n_jobs=NCORES)(
        delayed(compute_label)(scenario, x, **simu_kw) for x in X
    )
    t = timer() - t
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, t


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    n_samples = int(sys.argv[2])
    T = int(sys.argv[3])
    scenario_data = import_scenario_data(path)
    scenario = load_scenario(scenario_data)
    X, y, t = generate_dataset(scenario, n_samples, duration=T, timestep=1/FPS)
    print("Invalid", "Successes", "Failures", "Time")
    print((y == 0).sum(), (y == 1).sum(), (y == -1).sum(), t)


if __name__ == "__main__":
    main()
