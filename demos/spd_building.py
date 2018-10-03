import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from joblib import Memory
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.abspath(".."))
from core.robustness import (find_successful_samples_adaptive,  # noqa: E402
                             learn_success_probability)
from core.scenario import load_scenario  # noqa: E402
from gui.viewers import ScenarioViewer  # noqa: E402

memory = Memory(cachedir=".cache")


@memory.cache
def learn_proba(scenario_data):
    scenario = load_scenario(scenario_data)
    print("Number of dimensions:", len(scenario.design_space))
    np.random.seed(123)
    init_samples, init_labels = find_successful_samples_adaptive(
        scenario, n_succ=100, n_0=100, n_k=10, k_max=100, sigma=.01,
        duration=3, timestep=1/500
    )
    estimator = learn_success_probability(
        scenario, init_samples, init_labels, accuracy=.9, n_k=50, k_max=10,
        duration=3, timestep=1/500
    )
    return estimator


def visualize_proba(ax, estimator):
    n = 300
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    # Get probabilities
    X = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size)*.5))
    proba = estimator.predict_proba(X)[:, 1].reshape(xx.shape)
    # proba = estimator.decision_function(X).reshape(xx.shape)
    # Set colormap
    blue = np.array([31, 120, 180]) / 255
    green = np.array([178, 223, 138]) / 255
    cm = LinearSegmentedColormap.from_list('custom', [blue, green], N=10)
    # Plot
    m = ax.pcolormesh(xx, yy, proba, cmap=cm)
    m.set_rasterized(True)


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    with open(path, 'r') as f:
        scenario_data = json.load(f)
    estimator = learn_proba(scenario_data)
    # Make the figure.
    seaborn.set()
    fig, ax = plt.subplots()
    visualize_proba(ax, estimator)
    fig.tight_layout()
    # Interactive point picker
    clicked = []

    def onpick(event):
        if event.inaxes == ax:
            clicked.append([event.xdata, event.ydata])
            ax.scatter(event.xdata, event.ydata, color='k')
            fig.canvas.draw()
    fig.canvas.mpl_connect('button_release_event', onpick)
    plt.show()

    for c in clicked:
        scenario = load_scenario(scenario_data)
        instance = scenario.instantiate_from_sample(c + [.5], geom='HD')
        app = ScenarioViewer(instance, frame_rate=240)
        try:
            app.run()
        except SystemExit:
            app.destroy()


if __name__ == "__main__":
    main()
