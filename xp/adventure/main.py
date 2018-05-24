import os
import sys

import numpy as np
from joblib import Memory, dump, load

sys.path.insert(0, os.path.abspath("../.."))
from xp.adventure.scenarios import TeapotAdventure  # noqa: E402
from xp.robustness import ScenarioRobustnessEstimator  # noqa: E402
from xp.simulate import Simulation  # noqa: E402

memory = Memory(cachedir=".cache")


@memory.cache
def search_random_solution(n_cand=200):
    candidates = TeapotAdventure.sample_valid(n_cand, max_trials=3*n_cand,
                                              rule='R')

    def run_and_check(x):
        scenario = TeapotAdventure(x)
        simu = Simulation(scenario)
        simu.run()
        return scenario.succeeded()

    solution = next(cand for cand in candidates if run_and_check(cand))
    print(run_and_check(solution))
    return solution


def search_most_robust_solution(estimator: ScenarioRobustnessEstimator):
    candidates = TeapotAdventure.sample_valid(1000, max_trials=3000, rule='R')
    robustnesses = estimator.eval(candidates)
    return candidates[np.argmax(robustnesses)]


def view_solution(x):
    scenario = TeapotAdventure(x, make_geom=True)
    simu = Simulation(scenario, timestep=1/500)
    simu.run_visual(grid='xz', view_h=180)


def export(x, name):
    scenario = TeapotAdventure(x, make_geom=True)
    scenario.export_scene_to_pdf(name, x, (3*21, 2*29.7))


def main():
    x_manual = [
        -.30, .11, 11,      # top track
        -11,                # left track 1
        -.32, -.11, 10,     # left track 2
        -.22, -.16, -22,    # left track 3
        -.24, -.25, 15,     # left track 4
        15,                 # right track 1
        .33, -.12, -2,      # right track 2
        .25, -.18, 2,       # right track 3
        .32, -.27, -20,     # right track 4
        -.41, -.11,         # left pulley weight
        .43, -.10,          # right pulley weight
        -.06, .16,          # top pulley weight
        -.05, -.30,         # bottom pulley track
        -.05, -.50,         # bottom goblet
        .13,                # teapot x
        .28,                # top pulley p1 & p2 y
        -.04,               # left pulley p1 & p2 y
        0.,                 # right pulley p1 y
        .5                  # right pulley p2 x
    ]
    view_solution(x_manual)
    # x_random = search_random_solution()
    # view_solution(x_random)
    return
    filename = "full-robustness.pkl"
    try:
        full_rob_estimator = load(filename)
    except FileNotFoundError:
        full_rob_estimator = ScenarioRobustnessEstimator(TeapotAdventure)
        full_rob_estimator.train(n_samples=2000, verbose=True)
        dump(full_rob_estimator, filename)
    x_full_rob = search_most_robust_solution(full_rob_estimator)
    view_solution(x_full_rob)
    export(x_full_rob, filename[:-3] + "pdf")


if __name__ == "__main__":
    main()
