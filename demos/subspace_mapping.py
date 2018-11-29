import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory

sys.path.insert(0, os.path.abspath(".."))
import core.robustness as rob  # noqa: E402
from core.scenario import import_scenario_data, load_scenario  # noqa: E402

memory = Memory(cachedir=".cache")


@memory.cache
def initialize(scenario_data, n_succ, n_0, n_k, **simu_kw):
    #
    scenario = load_scenario(scenario_data)
    return rob.find_successful_samples_adaptive(
        scenario, n_succ=n_succ, n_0=n_0, n_k=n_k, k_max=500, sigma=.01,
        ret_event_labels=True, **simu_kw
    )


def plot_mapping(assignments, dim_names):
    key_ev, key_ass = zip(*[it for it in assignments.items() if len(it[1])])
    ass_matrix = np.zeros((len(dim_names), len(key_ev)), dtype=bool)
    for i, ass in enumerate(key_ass):
        ass_matrix[:, i][ass] = True
    fig, ax = plt.subplots()
    ax.matshow(ass_matrix, aspect='auto', cmap=plt.cm.Reds)
    ax.set_xticks(range(len(key_ev)))
    ax.set_xticklabels(key_ev, rotation=15, rotation_mode='anchor',
                       ha='left')
    ax.set_yticks(range(len(dim_names)))
    ax.set_yticklabels(dim_names)
    fig.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    scenario_data = import_scenario_data(path)
    if scenario_data is None:
        return
    np.random.seed(111)
    duration = 4
    timestep = 1 / 500

    # Initial exploration
    n_succ = 300
    n_0 = 200
    n_k = 30
    init_samples, _, init_event_labels = initialize(
        scenario_data, n_succ, n_0, n_k, duration=duration, timestep=timestep
    )

    # Subspace mapping
    assignments = rob.map_events_to_dimensions(init_samples, init_event_labels,
                                               select_coeff=.2)
    dim_names = load_scenario(scenario_data).design_space.free_parameters_names
    plot_mapping(assignments, dim_names)


if __name__ == "__main__":
    main()
