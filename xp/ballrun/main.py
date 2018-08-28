import os
import sys

import numpy as np
from joblib import Memory, dump, load

sys.path.insert(0, os.path.abspath("../.."))
from xp.ifelse.scenarios import ConditionalBallRun  # noqa: E402
from xp.robustness import FullScenarioRobustnessEstimator  # noqa: E402
from xp.simulate import Simulation  # noqa: E402

memory = Memory(cachedir=".cache")


@memory.cache
def search_random_solution(n_cand=200):
    candidates = ConditionalBallRun.sample_valid(n_cand, max_trials=3*n_cand,
                                                 rule='R')

    def run_and_check(x):
        scenario = ConditionalBallRun(x)
        simu = Simulation(scenario)
        simu.run()
        return scenario.succeeded()

    solution = next(cand for cand in candidates if run_and_check(cand))
    print(run_and_check(solution))
    return solution


def search_most_robust_solution(estimator: FullScenarioRobustnessEstimator):
    candidates = ConditionalBallRun.sample_valid(1000, max_trials=3000,
                                                 rule='R')
    robustnesses = estimator.eval(candidates)
    return candidates[np.argmax(robustnesses)]


def view_solution(x):
    scenario = ConditionalBallRun(x, make_geom=True)
    simu = Simulation(scenario, timestep=1/500)
    simu.run_visual()


def export(x, name):
    scenario = ConditionalBallRun(x, make_geom=True)
    scenario.export_scene_to_pdf(name, x, (3*21, 2*29.7))


def main():
    # x_manual = [0.1, 0.3, -20, 0.1, 10, 0.35, -0.05, -45]
    x_random = search_random_solution()
    view_solution(x_random)
    filename = "full-robustness.pkl"
    try:
        full_rob_estimator = load(filename)
    except FileNotFoundError:
        full_rob_estimator = FullScenarioRobustnessEstimator(
            ConditionalBallRun
        )
        full_rob_estimator.train(n_samples=2000, verbose=True)
        dump(full_rob_estimator, filename)
    x_full_rob = search_most_robust_solution(full_rob_estimator)
    view_solution(x_full_rob)
    export(x_full_rob, filename[:-3] + "pdf")


if __name__ == "__main__":
    main()
