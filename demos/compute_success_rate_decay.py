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
from bayes_opt import BayesianOptimization
from joblib import delayed, Memory, Parallel
from scipy.stats import sem
from timeit import default_timer as timer

sys.path.insert(0, os.path.abspath(".."))
import core.robustness as rob  # noqa: E402
from core.scenario import (import_scenario_data, load_scenario)  # noqa: E402

SIMU_KW = dict(timestep=1/500, duration=0)
NCORES = 2
memory = Memory(cachedir=".cache")


# Parallelized
def compute_label(scenario, x):
    if scenario.check_physically_valid_sample(x):
        label = 2*int(rob.compute_label(scenario, x, **SIMU_KW)) - 1
    else:
        label = 0
    return label


# Memoized
@memory.cache
def generate_dataset(scenario, n_samples):
    t = timer()
    ndims = len(scenario.design_space)
    X = rob.MultivariateUniform(ndims).sample(n_samples)
    # labels = [compute_label(scenario, x) for x in X]
    y = Parallel(n_jobs=NCORES)(delayed(compute_label)(scenario, x) for x in X)
    t = timer() - t
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, t


def compute_success_rates(X, y, center, radii):
    valid = np.flatnonzero(y)
    y = y[valid]
    d = np.linalg.norm(X[valid] - center, axis=1)
    return [(y[d <= r] == 1).mean() for r in radii]


class LocalRobustnessEstimator:
    def __init__(self, scenario, radius, n_neighbors):
        self.scenario = scenario
        self.radius = radius
        self.n_neighbors = n_neighbors

    def __call__(self, x):
        x = np.asarray(x)
        scenario = self.scenario
        if not scenario.check_physically_valid_sample(x):
            return 0.
        radius = self.radius
        n_neighbors = self.n_neighbors
        dist = rob.MultivariateUniform(x.size, x-radius, x+radius)
        X = rob.find_physically_valid_samples(
            scenario, dist, n_neighbors, 100*n_neighbors
        )
        X.append(x)
        y = [rob.compute_label(scenario, xi, **SIMU_KW) for xi in X]
        return sum(y) / len(y)

    def from_dict(self, **x_dict):
        return self(self.dict2array(x_dict))

    @staticmethod
    def dict2array(x_dict):
        x = np.empty(len(x_dict))
        for k, v in x_dict.items():
            x[int(k)] = v
        return x


# Memoized
@memory.cache
def find_best_uniform(rob_est, n_eval):
    n_dims = len(rob_est.scenario.design_space)
    dist = rob.MultivariateUniform(n_dims)
    X = rob.find_physically_valid_samples(
        rob_est.scenario, dist, n_eval, 100*n_eval
    )
    r = [rob_est(x) for x in X]
    x_best = X[np.argmax(r)]
    return x_best


# Memoized
@memory.cache
def find_best_gpo(rob_est, n_eval):
    n_dims = len(rob_est.scenario.design_space)
    pbounds = {str(i): (0, 1) for i in range(n_dims)}
    optimizer = BayesianOptimization(f=rob_est.from_dict, pbounds=pbounds)
    n_init = n_eval // 2
    n_iter = n_eval - n_init
    optimizer.maximize(init_points=n_init, n_iter=n_iter, acq='ei', xi=0.)
    return rob_est.dict2array(optimizer.max['params'])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    n_samples = int(sys.argv[2])
    SIMU_KW['duration'] = int(sys.argv[3])
    scenario_data = import_scenario_data(path)
    scenario = load_scenario(scenario_data)
    X, y, t = generate_dataset(scenario, n_samples)
    print("Invalid", "Successes", "Failures", "Time")
    print((y == 0).sum(), (y == 1).sum(), (y == -1).sum(), t)

    n_err = 20
    errors = np.linspace(.01, .5, n_err)
    budget = 1000  # number of simulations for each method
    radius = .1  # radius of the ball to compute local robustness
    n_neighbors = 49  # number of neighbors to simulate to compute local rob
    n_eval = budget // (n_neighbors + 1)
    rob_est = LocalRobustnessEstimator(scenario, radius, n_neighbors)
    fig, ax = plt.subplots()

    # Compute the average decay of all successes.
    valid = np.flatnonzero(y)[:budget]
    successes = valid[y[valid] == 1]
    decays = np.array([compute_success_rates(X, y, x, errors)
                       for x in X[successes]])
    avg_decay = np.mean(decays, axis=0)
    sem_decay = sem(decays, axis=0)
    ax.plot(errors, avg_decay, color='blue',
            label="Average success (n={})".format(successes.size))
    ax.fill_between(errors, avg_decay - sem_decay, avg_decay + sem_decay,
                    color='blue', alpha=.5)

    # Compute the decay of the most robust success (uniform search).
    x_uni = find_best_uniform(rob_est, n_eval)
    X_uni = np.vstack((X, x_uni))
    y_uni = np.append(y, compute_label(scenario, x_uni))
    decay = compute_success_rates(X_uni, y_uni, x_uni, errors)
    ax.plot(errors, decay, color='red', label="Most robust uniform")

    # Compute the decay of the most robust success (GP optimization).
    x_gpo = find_best_gpo(rob_est, n_eval)
    X_gpo = np.vstack((X, x_gpo))
    y_gpo = np.append(y, compute_label(scenario, x_gpo))
    decay = compute_success_rates(X_gpo, y_gpo, x_gpo, errors)
    ax.plot(errors, decay, color='purple', label="Most robust GPO")

    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
