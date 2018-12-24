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

import matplotlib.pyplot as plt
import numpy as np
from joblib import delayed, Memory, Parallel
from scipy.stats import sem
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


def compute_success_rates(X, y, center, errors):
    success_rates = []
    for e in errors:
        d = np.linalg.norm(X - center, axis=1)
        y_in_ball = y[d <= e]
        sr = (y_in_ball == 1).sum() / (y_in_ball != 0).sum()
        success_rates.append(sr)
    return success_rates


# Memoized
@memory.cache
def compute_robust_random(scenario, X, y, budget, radius, **simu_kw):
    # Use half of the budget to simulate valid samples (reuse X to save time).
    X_val = X[y != 0][:budget // 2]
    y_val = y[y != 0][:budget // 2]
    X_succ = X_val[y_val == 1]
    # Use the other half to evaluate the robustness of each success.
    robs = np.empty(X_succ.shape[0])
    n_neighbors = (budget - budget//2) // X_succ.shape[0]
    ndims = len(scenario.design_space)
    for i, x in enumerate(X_succ):
        dist = rob.MultivariateUniform(ndims, x-radius, x+radius)
        X_neighbors = rob.find_physically_valid_samples(
            scenario, dist, n_neighbors, 100*n_neighbors
        )
        y_neighbors = [
            rob.compute_label(scenario, x, **simu_kw)
            for x in X_neighbors
        ]
        robs[i] = sum(y_neighbors) / n_neighbors
    x_best = X_succ[np.argmax(robs)]
    return x_best


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

    n_err = 20
    errors = np.linspace(.01, .5, n_err)
    fig, ax = plt.subplots()

    # Compute the decay of a set of random successes.
    n_rs = 10
    X_rs = X[np.random.choice(np.flatnonzero(y == 1), n_rs, replace=False)]
    decays = np.asarray([compute_success_rates(X, y, x, errors) for x in X_rs])
    avg_decay = np.mean(decays, axis=0)
    sem_decay = sem(decays, axis=0)
    ax.plot(errors, avg_decay, color='blue',
            label="Random (n={})".format(n_rs))
    ax.fill_between(errors, avg_decay - sem_decay, avg_decay + sem_decay,
                    color='blue', alpha=.5)

    # Compute the decay of a robust random success.
    radius = .1  # radius of the ball to compute robustness
    budget = 1000  # number of simulations for the entire method
    x_rr = compute_robust_random(scenario, X, y, budget, radius,
                                 duration=T, timestep=1/FPS)
    decay = compute_success_rates(X, y, x_rr, errors)
    ax.plot(errors, decay, color='red', label="Robust random")

    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
