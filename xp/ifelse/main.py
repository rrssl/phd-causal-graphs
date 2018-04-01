import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
from xp.ifelse.scenarios import ConditionalBallRun  # noqa: E402
from xp.simulate import Simulation  # noqa: E402


def main():
    x_init = [0.1, 0.3, -20, 0.1, 10, 0.35, -0.05, -45]
    # x_init = ConditionalBallRun.sample_valid(9)[6]
    scenario = ConditionalBallRun(x_init, make_geom=True)
    simu = Simulation(scenario)
    simu.run_visual()
    scenario.export_scene_to_pdf("test", x_init, (3*21, 2*29.7))


if __name__ == "__main__":
    main()
