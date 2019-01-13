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
import seaborn
from bayes_opt import BayesianOptimization, UtilityFunction
from joblib import delayed, Memory, Parallel
from scipy.stats import sem
from timeit import default_timer as timer
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(".."))
import core.optimize as opt  # noqa: E402
import core.robustness as rob  # noqa: E402
from core.scenario import import_scenario_data, load_scenario  # noqa: E402
from core.config import NCORES  # noqa: E402

SIMU_KW = dict(timestep=1/500, duration=0)
memory = Memory(cachedir=".cache")


# Parallelizable
def compute_label(scenario, x):
    if scenario.check_physically_valid_sample(x):
        label = 2*int(rob.compute_label(scenario, x, **SIMU_KW)) - 1
    else:
        label = 0
    return label


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


@memory.cache
def generate_dataset2(scenario, n_samples):
    t = timer()
    n_dims = len(scenario.design_space)
    X = rob.SobolSequence(n_dims).sample(n_samples)
    # labels = [compute_label(scenario, x) for x in X]
    y = Parallel(n_jobs=NCORES)(delayed(compute_label)(scenario, x)
                                for x in tqdm(X))
    t = timer() - t
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, t


def compute_success_rates(X, y, center, radii):
    valid = np.flatnonzero(y)
    y = y[valid]
    d = np.linalg.norm(X[valid] - center, axis=1)
    sr = np.zeros(len(radii))
    for i, r in enumerate(radii):
        y_in_ball = y[d <= r]
        if y_in_ball.size:
            sr[i] = (y_in_ball == 1).mean()
    return sr


class LocalRobustnessEstimator:
    def __init__(self, scenario, radius, n_neighbors):
        self.scenario = scenario
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.n_eval = 0

    def __call__(self, x):
        x = np.asarray(x)
        scenario = self.scenario
        if not scenario.check_physically_valid_sample(x):
            return 0.
        self.n_eval += 1
        radius = self.radius
        n_neighbors = self.n_neighbors
        dist = rob.MultivariateUniform(x.size, x-radius, x+radius)
        X = rob.find_physically_valid_samples(
            scenario, dist, n_neighbors, 100*n_neighbors
        )
        X.append(x)
        y = Parallel(n_jobs=NCORES)(
            delayed(rob.compute_label)(scenario, xi, **SIMU_KW) for xi in X
        )
        return sum(y) / len(y)

    def from_dict(self, **x_dict):
        return self(self.dict2array(x_dict))

    @staticmethod
    def dict2array(x_dict):
        x = np.empty(len(x_dict))
        for k, v in x_dict.items():
            x[int(k)] = v
        return x


@memory.cache
def find_best_uniform(rob_est, n_eval, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_dims = len(rob_est.scenario.design_space)
    dist = rob.MultivariateUniform(n_dims)
    X = rob.find_physically_valid_samples(
        rob_est.scenario, dist, n_eval, 100*n_eval
    )
    r = [rob_est(x) for x in X]
    x_best = X[np.argmax(r)]
    return x_best


@memory.cache
def find_best_gpo(rob_est, n_eval, xi=0., acq='ei', seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_dims = len(rob_est.scenario.design_space)
    pbounds = {str(i): (0, 1) for i in range(n_dims)}
    optimizer = BayesianOptimization(rob_est.from_dict, pbounds, seed)
    # We compute our own initial valid points.
    n_init = n_eval // 2
    dist = rob.MultivariateUniform(n_dims)
    X_init = rob.find_physically_valid_samples(
        rob_est.scenario, dist, n_init, 100*n_init
    )
    for x in X_init:
        optimizer.probe({str(i): x[i] for i in range(n_dims)})
    # Run main maximization routine.
    n_iter = n_eval - n_init
    optimizer.maximize(init_points=0, n_iter=n_iter, acq=acq, xi=xi)
    # Add iterations until n_eval has effectively been spent.
    util = UtilityFunction(kind=acq, kappa=2.576, xi=xi)  # as in maximize()
    while rob_est.n_eval < n_eval:
        optimizer.probe(optimizer.suggest(util), lazy=False)
    return rob_est.dict2array(optimizer.max['params'])


@memory.cache
def find_successful_samples_adaptive(scenario, n_succ, n_0, n_k, k_max, sigma,
                                     seed=None):
    # 'seed' is just here to cache several results
    #
    return rob.find_successful_samples_adaptive(
        scenario, n_succ, n_0, n_k, k_max, sigma, **SIMU_KW
    )


@memory.cache
def train_and_consolidate_boundary(scenario, X, y, accuracy, n_k, k_max,
                                   ret_simu_cost=False):
    #
    step_data = [] if ret_simu_cost else None
    rob_est = rob.train_and_consolidate_boundary2(
        scenario, X, y, accuracy, n_k, k_max, step_data=step_data, **SIMU_KW
    )
    if ret_simu_cost:
        simu_cost = len(step_data[-1][0])
        return rob_est, simu_cost
    else:
        return rob_est


@memory.cache
def find_best_ours(scenario, active=True, optimizer='global', seed=None,
                   ret_simu_cost=False):
    if seed is not None:
        np.random.seed(seed)
    # Initialize
    #
    X, y = find_successful_samples_adaptive(
        scenario, n_succ=100, n_0=500, n_k=10, k_max=500, sigma=.01, seed=seed
    )
    if ret_simu_cost:
        simu_cost = len(y)
    # Build the robustness estimator
    if active:
        if ret_simu_cost:
            rob_est, simu_cost = train_and_consolidate_boundary(
                scenario, X, y, accuracy=1, n_k=10, k_max=5,
                ret_simu_cost=True
            )
        else:
            rob_est = train_and_consolidate_boundary(
                scenario, X, y, accuracy=1, n_k=10, k_max=5,
                ret_simu_cost=False
            )
    else:
        rob_est = rob.train_svc(X, y, probability=True, verbose=False)
    # Find optimal solution
    init_probas = rob_est.predict_proba(X)[:, 1]
    if optimizer is None:
        x = X[np.argmax(init_probas)]
    elif optimizer == 'local':
        x0 = X[np.argmax(init_probas)]
        res = opt.maximize_robustness_local(scenario, [rob_est], x0)
        x = res.x
    elif optimizer == 'global':
        # init = np.asarray(X)[np.argpartition(init_probas, -100)[-100:]]
        x0 = X[np.argmax(init_probas)]
        res = opt.maximize_robustness_global(scenario, [rob_est], x0,
                                             fevals=100, **SIMU_KW)
        x = res[0]
        simu_cost += res[3]
    if ret_simu_cost:
        return x, simu_cost
    else:
        return x


@memory.cache
def compute_ours(X, y, errors, n_runs, scenario, active=True,
                 optimizer='local', trial=0):
    curves = []
    simu_costs = []
    for i in range(n_runs):
        x_ours, simu_cost = find_best_ours(
            scenario, active=active, optimizer=optimizer, seed=i,
            ret_simu_cost=True
        )
        X_ours = np.vstack((X, x_ours))
        y_ours = np.append(y, compute_label(scenario, x_ours))
        curve = compute_success_rates(X_ours, y_ours, x_ours, errors)
        curves.append(curve)
        simu_costs.append(simu_cost)
    avg_curve = np.mean(curves, axis=0)
    sem_curve = sem(curves, axis=0)
    simu_cost = sum(simu_costs) // len(simu_costs)
    return avg_curve, sem_curve, simu_cost


@memory.cache
def compute_B1(X, y, errors, n_runs):
    successes = np.flatnonzero(y == 1)[:n_runs]
    curves = np.array([compute_success_rates(X, y, x, errors)
                       for x in X[successes]])
    avg_curve = np.mean(curves, axis=0)
    sem_curve = sem(curves, axis=0)
    return avg_curve, sem_curve


@memory.cache
def compute_B2(X, y, errors, n_runs, n_eval, rob_est, trial=0):
    curves = []
    for i in range(n_runs):
        rob_est.n_eval = 0
        x_uni = find_best_uniform(rob_est, n_eval, seed=i)
        X_uni = np.vstack((X, x_uni))
        y_uni = np.append(y, compute_label(rob_est.scenario, x_uni))
        curve = compute_success_rates(X_uni, y_uni, x_uni, errors)
        curves.append(curve)
    avg_curve = np.mean(curves, axis=0)
    sem_curve = sem(curves, axis=0)
    return avg_curve, sem_curve


@memory.cache
def compute_B3(X, y, errors, n_runs, n_eval, rob_est, trial=0):
    curves = []
    for i in range(n_runs):
        rob_est.n_eval = 0
        x_gpo = find_best_gpo(rob_est, n_eval, seed=i)
        X_gpo = np.vstack((X, x_gpo))
        y_gpo = np.append(y, compute_label(rob_est.scenario, x_gpo))
        curve = compute_success_rates(X_gpo, y_gpo, x_gpo, errors)
        curves.append(curve)
    avg_curve = np.mean(curves, axis=0)
    sem_curve = sem(curves, axis=0)
    return avg_curve, sem_curve


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    n_samples = int(sys.argv[2])
    SIMU_KW['duration'] = int(sys.argv[3])
    scenario_data = import_scenario_data(path)
    scenario = load_scenario(scenario_data)
    # X, y, t = generate_dataset(scenario, n_samples)
    X, y, t = generate_dataset2(scenario, n_samples)
    print("Invalid", "Successes", "Failures", "Time")
    print((y == 0).sum(), (y == 1).sum(), (y == -1).sum(), t)

    n_dims = X.shape[1]
    n_err = 30  # number of errors to build the success rate plot
    errors = np.linspace(0, 1, n_err)
    n_runs = 10  # number of runs for each random method
    seaborn.set()
    fig, ax = plt.subplots(figsize=(6, 2))

    # # Compute the decay of the most robust solution of our passive method.
    # avg_curve, sem_curve, simu_cost = compute_ours(
    #     X, y, errors, n_runs, scenario, active=False, optimizer=None
    # )
    # ax.plot(errors, avg_curve,
    #         label="Most robust: ours, passive, noopt (n={})".format(n_runs))
    # ax.fill_between(errors, avg_curve - sem_curve, avg_curve + sem_curve,
    #                 alpha=.5)

    # # Compute the decay of the most robust solution of our active method.
    # avg_curve, sem_curve, simu_cost = compute_ours(
    #     X, y, errors, n_runs, scenario, active=True, optimizer=None
    # )
    # ax.plot(errors, avg_curve,
    #         label="Most robust: ours, active, noopt (n={})".format(n_runs))
    # ax.fill_between(errors, avg_curve - sem_curve, avg_curve + sem_curve,
    #                 alpha=.5)

    # Compute the decay of the most robust solution of our method.
    avg_curve, sem_curve, simu_cost = compute_ours(
        X, y, errors, n_runs, scenario, active=True, optimizer='local', trial=0
    )
    ax.plot(errors, avg_curve, label="Ours")
    ax.fill_between(errors, avg_curve - sem_curve, avg_curve + sem_curve,
                    alpha=.5)

    # simu_budget = 1000  # number of simus allowed for each method
    simu_budget = simu_cost
    print("Number of simulations allowed:", simu_budget)

    # B1: random uniform successes.
    avg_curve, sem_curve = compute_B1(X, y, errors, n_runs)
    ax.plot(errors, avg_curve, label="B1")
    ax.fill_between(errors, avg_curve - sem_curve, avg_curve + sem_curve,
                    alpha=.5)

    # Initialize options for baselines based on local robustness evaluation.
    radius = .1  # radius of the ball to compute local robustness
    n_neighbors = n_dims**2  # number of simulations to compute local rob
    n_eval = simu_budget // (n_neighbors + 1)  # number of rob eval allowed
    rob_est = LocalRobustnessEstimator(scenario, radius, n_neighbors)

    # B2: robustness-based uniform search.
    avg_curve, sem_curve = compute_B2(X, y, errors, n_runs, n_eval, rob_est)
    ax.plot(errors, avg_curve, label="B2")
    ax.fill_between(errors, avg_curve - sem_curve, avg_curve + sem_curve,
                    alpha=.5)

    # B3: Bayesian optimization.
    avg_curve, sem_curve = compute_B3(X, y, errors, n_runs, n_eval, rob_est)
    ax.plot(errors, avg_curve, label="B3")
    ax.fill_between(errors, avg_curve - sem_curve, avg_curve + sem_curve,
                    alpha=.5)

    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
