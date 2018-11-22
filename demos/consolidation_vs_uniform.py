import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from joblib import Memory
from sklearn import metrics

sys.path.insert(0, os.path.abspath(".."))
import core.robustness as rob  # noqa: E402
from core.scenario import import_scenario_data, load_scenario  # noqa: E402

memory = Memory(cachedir=".cache")


@memory.cache
def create_reference_dataset(scenario_data, n, **simu_kw):
    scenario = load_scenario(scenario_data)
    ndims = len(scenario.design_space)
    samples = rob.find_physically_valid_samples(
        scenario, rob.MultivariateUniform(ndims), n, 100*n
    )
    print("Number of samples:", len(samples))
    labels = [rob._simulate_and_get_success(scenario, s, **simu_kw)
              for s in samples]
    return np.asarray(samples), np.asarray(labels)


@memory.cache
def compute_reference_spd(samples, labels):
    return rob.train_svc(samples, labels, probability=True)


@memory.cache
def initialize(scenario_data, n_succ=100, n_0=50, n_k=10, **simu_kw):
    scenario = load_scenario(scenario_data)
    return rob.find_successful_samples_adaptive(
        scenario, n_succ=n_succ, n_0=n_0, n_k=n_k, k_max=500, sigma=.01,
        **simu_kw
    )


@memory.cache
def run_uniform(scenario_data, init_samples, init_labels, n_k, k_max):
    scenario = load_scenario(scenario_data)
    print("Number of dimensions:", len(scenario.design_space))
    step_data = []
    rob.train_and_add_uniform_samples(
        scenario, init_samples, init_labels, accuracy=1, n_k=n_k, k_max=k_max,
        step_data=step_data, duration=3, timestep=1/500
    )
    return step_data


@memory.cache
def run_consol(scenario_data, init_samples, init_labels, n_k, k_max):
    #
    scenario = load_scenario(scenario_data)
    print("Number of dimensions:", len(scenario.design_space))
    step_data = []
    rob.train_and_consolidate_boundary(
        scenario, init_samples, init_labels, accuracy=1, n_k=n_k, k_max=k_max,
        step_data=step_data, duration=3, timestep=1/500
    )
    return step_data


@memory.cache
def run_consol2(scenario_data, init_samples, init_labels, n_k, k_max):
    #
    scenario = load_scenario(scenario_data)
    print("Number of dimensions:", len(scenario.design_space))
    step_data = []
    rob.train_and_consolidate_boundary2(
        scenario, init_samples, init_labels, accuracy=1, n_k=n_k, k_max=k_max,
        step_data=step_data, duration=3, timestep=1/500
    )
    return step_data


@memory.cache
def process_data(step_data, test_set):
    num_samples = []
    scores = []
    for X, y, estimator in step_data:
        num_samples.append(len(X))
        # est_score = estimator.score(X, y)
        # est_score = estimator.score(*test_set)
        est_score = metrics.balanced_accuracy_score(
            test_set[1], estimator.predict(test_set[0])
        )
        scores.append(est_score)
    # return np.arange(len(step_data)), scores
    return num_samples, scores


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    scenario_data = import_scenario_data(path)
    if scenario_data is None:
        return
    np.random.seed(111)
    # Create reference dataset.
    test_set = create_reference_dataset(scenario_data, 10000,
                                        duration=3, timestep=1/500)
    nts = sum(test_set[1])
    ntf = len(test_set[1]) - nts
    print("The test set had {} failures and {} successes.".format(ntf, nts))
    # ref_spd = compute_reference_spd(*test_set)
    # Run the algorithms.
    n_k = 50
    k_max = 10
    np.random.seed(222)
    init_samples, init_labels = initialize(
        scenario_data, duration=3, timestep=1/500
    )
    # Make the figure.
    seaborn.set()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ns_unif, acc_unif, 'b--', linewidth=3,
            label="uniform sampling")
    ax.plot(ns_cons, acc_cons, 'g', linewidth=3,
            label="boundary consolidation")
    # ax.plot(ns_cons2, acc_cons2, 'r', linewidth=3,
    #         label="boundary consolidation2")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
