import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
from xp.ifelse.scenarios import ConditionalBallRun, StateObserver  # noqa: E402
from xp.simulate import Simulation  # noqa: E402


def main():
    x_manual = [0.1, 0.3, -20, 0.1, 10, 0.35, -0.05, -45]
    scenario = ConditionalBallRun(x_manual, make_geom=True)
    observer = StateObserver(scenario._scene)

    scenario.export_scene_to_egg("scene.egg")
    simu = Simulation(scenario, [observer], timestep=1/500)
    simu.run()
    observer.export("states.pkl")


if __name__ == "__main__":
    main()
