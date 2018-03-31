import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
from xp.ifelse.scenarios import ConditionalBallRun  # noqa: E402
from xp.simulate import Simulation  # noqa: E402


def main():
    x_init = [0.1, 0.3, -20, 0.1, 10, 0.35, -0.05, -45]
    scenario = ConditionalBallRun(x_init, make_geom=True)
    simu = Simulation(scenario)
    simu.run_visual()


if __name__ == "__main__":
    main()
