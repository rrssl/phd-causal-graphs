import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn

sys.path.insert(0, os.path.abspath(".."))
from core.robustness import (find_successful_samples_adaptive,  # noqa: E402
                             find_successful_samples_uniform)
from core.scenario import load_scenario  # noqa: E402


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    with open(path, 'r') as f:
        scenario_data = json.load(f)
    scenario = load_scenario(scenario_data)
    print("Number of dimensions:", len(scenario.design_space))
    totals_u = []
    np.random.seed(123)
    find_successful_samples_uniform(scenario, n_succ=100, n_0=50, n_k=10,
                                    k_max=100, totals=totals_u,
                                    duration=3, timestep=1/500)
    totals_a = []
    np.random.seed(123)
    find_successful_samples_adaptive(scenario, n_succ=100, n_0=50, n_k=10,
                                     k_max=100, sigma=.01, totals=totals_a,
                                     duration=3, timestep=1/500)
    # Make the figure.
    seaborn.set()
    fig, ax = plt.subplots()
    ax.plot(totals_u, label="uniform")
    ax.plot(totals_a, label="adaptive")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
