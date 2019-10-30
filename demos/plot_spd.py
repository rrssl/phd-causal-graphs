import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from joblib import Memory
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.abspath(".."))
from core.robustness import (find_successful_samples_adaptive,  # noqa: E402
                             learn_success_probability)
from core.scenario import (StateObserver, import_scenario_data,  # noqa: E402
                           load_scenario)
from gui.viewers import Replayer  # noqa: E402

memory = Memory(cachedir=".cache")


@memory.cache
def learn_proba(scenario_data):
    scenario = load_scenario(scenario_data)
    print("Number of dimensions:", len(scenario.design_space))
    np.random.seed(123)
    init_samples, init_labels = find_successful_samples_adaptive(
        scenario, n_succ=200, n_0=300, n_k=20, k_max=100, sigma=.01,
        duration=3, timestep=1/500
    )
    estimator = learn_success_probability(
        scenario, init_samples, init_labels, accuracy=.95, n_k=100, k_max=10,
        duration=3, timestep=1/500
    )
    return estimator


def visualize_proba(ax, estimator, extra_params=None):
    n = 300
    xx, yy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    # Get probabilities
    samples = np.column_stack((xx.ravel(), yy.ravel()))
    if extra_params:
        samples = np.hstack((samples, np.tile(extra_params, (xx.size, 1))))
    proba = estimator.predict_proba(samples)[:, 1].reshape(xx.shape)
    # proba = estimator.decision_function(X).reshape(xx.shape)
    # Set colormap
    blue = np.array([31, 120, 180]) / 255
    green = np.array([178, 223, 138]) / 255
    cm = LinearSegmentedColormap.from_list('custom', [blue, green], N=10)
    # Plot
    m = ax.pcolormesh(xx, yy, proba, cmap=cm)
    m.set_rasterized(True)
    ax.figure.colorbar(m)


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    scenario_data = import_scenario_data(path)
    estimator = learn_proba(scenario_data)
    # Make the figure.
    seaborn.set()
    fig, ax = plt.subplots()
    extra_params = [.2]
    visualize_proba(ax, estimator, extra_params)
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

    dir_ = tempfile.mkdtemp()
    for i, c in enumerate(clicked):
        p = c + extra_params
        print("Sample: {}; probablity of success: {}".format(
            p, estimator.predict_proba([p])
        ))
        scenario = load_scenario(scenario_data)
        # Create the scene geometry.
        instance = scenario.instantiate_from_vector(p, geom='HD', phys=False)
        scene_path = os.path.join(dir_, "scene")
        instance.scene.export_scene_to_egg(scene_path)
        instance.scene.export_scene_to_egg("scene_{}".format(i))
        # Run the instance.
        instance = scenario.instantiate_from_vector(p, geom=None, phys=True)
        obs = StateObserver(instance.scene)
        print("Physically valid: ", instance.scene.check_physically_valid())
        instance.simulate(duration=3, timestep=1/500, callbacks=[obs])
        simu_path = os.path.join(dir_, "simu.pkl")
        obs.export(simu_path, fps=500)
        # Show the simulation.
        app = Replayer(scene_path+".bam", simu_path)
        try:
            app.run()
        except SystemExit:
            app.destroy()


if __name__ == "__main__":
    main()
