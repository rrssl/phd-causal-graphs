import os
import sys
import tempfile

import numpy as np
from joblib import Memory
from panda3d.core import load_prc_file_data

sys.path.insert(0, os.path.abspath(".."))
import core.robustness as rob  # noqa: E402
from core.optimize import maximize_robustness  # noqa: E402
from core.scenario import (StateObserver, import_scenario_data,  # noqa: E402
                           load_scenario, simulate_scene)
from gui.viewers import Replayer, ScenarioViewer  # noqa: E402

memory = Memory(cachedir=".cache")


@memory.cache
def initialize(scenario_data, **simu_kw):
    scenario = load_scenario(scenario_data)
    return rob.find_successful_samples_adaptive(
        scenario, n_succ=100, n_0=50, n_k=10, k_max=500, sigma=.01, **simu_kw
    )


@memory.cache
def compute_rob(scenario_data, init_samples, init_labels, n_k, k_max,
                **simu_kw):
    scenario = load_scenario(scenario_data)
    print("Number of dimensions:", len(scenario.design_space))
    return rob.train_and_consolidate_boundary(
        scenario, init_samples, init_labels, accuracy=.9, n_k=n_k, k_max=k_max,
        **simu_kw
    )


@memory.cache
def optimize_rob(scenario_data, estimators, x0, smin_coeff):
    scenario = load_scenario(scenario_data)
    res = maximize_robustness(scenario, estimators, x0, smin_coeff)
    return res.x


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    scenario_data = import_scenario_data(path)
    if scenario_data is None:
        return
    # Run the algorithms.
    n_k = 50
    k_max = 10
    duration = 4
    timestep = 1 / 500
    smin_coeff = .1
    np.random.seed(111)
    init_samples, init_labels = initialize(
        scenario_data, duration=duration, timestep=timestep
    )
    estimator = compute_rob(
        scenario_data, init_samples, init_labels, n_k, k_max,
        duration=duration, timestep=timestep
    )
    x_init = init_samples[
        np.argmax(estimator.predict_proba(init_samples)[:, 1])
    ]
    x_best = optimize_rob(scenario_data, [estimator], x_init, smin_coeff)

    scenario = load_scenario(scenario_data)
    if 0:
        # Print the solution.
        instance = scenario.instantiate_from_sample(
            x_best, geom='LD', phys=True
        )
        instance.scene.export_layout_to_pdf(
            "right_faster_yz", (21, 29.7), plane='yz', exclude="ground_geom",
            flip_v=True
        )
    if 0:
        # Show the solution.
        load_prc_file_data("", "win-origin 500 200")
        instance = scenario.instantiate_from_sample(
            x_best, geom='LD', phys=True
        )
        app = ScenarioViewer(instance)
        app.run()
    if 0:
        dir_ = tempfile.mkdtemp()
        # Run the instance.
        instance = scenario.instantiate_from_sample(
            x_best, geom='HD', phys=False
        )
        scene_path = os.path.join(dir_, "scene")
        instance.scene.export_scene_to_egg(scene_path)
        instance = scenario.instantiate_from_sample(
            x_best, geom=None, phys=True
        )
        obs = StateObserver(instance.scene)
        simu_path = os.path.join(dir_, "simu.pkl")
        simulate_scene(instance.scene, duration=duration, timestep=timestep,
                       callbacks=[obs])
        obs.export(simu_path, fps=int(1/timestep))
        # Show the simulation.
        app = Replayer(scene_path+".bam", simu_path)
        app.run()
    if 1:
        instance = scenario.instantiate_from_sample(
            x_best, geom='HD', phys=True
        )
        instance.scene.export_scene_to_egg("scene.egg")
        obs = StateObserver(instance.scene)
        simulate_scene(instance.scene, duration=duration, timestep=timestep,
                       callbacks=[obs])
        obs.export("simu.pkl", fps=int(1/timestep))


if __name__ == "__main__":
    main()
