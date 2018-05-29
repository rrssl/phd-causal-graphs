import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory, dump, load, Parallel, delayed

sys.path.insert(0, os.path.abspath("../.."))
from gui.viewers import Replayer  # noqa: E402
import xp.adventure.config as cfg  # noqa: E402
from xp.adventure.scenarios import StateObserver, TeapotAdventure  # noqa: E402
from xp.causal import EventState  # noqa: E402
from xp.robustness import ScenarioRobustnessEstimator  # noqa: E402
from xp.simulate import Simulation  # noqa: E402

memory = Memory(cachedir=".cache")


def run_and_check(x):
    scenario = TeapotAdventure(x)
    simu = Simulation(scenario)
    simu.run()
    if scenario.succeeded():
        return True, None
    else:
        events = scenario.causal_graph.get_events()
        try:
            failure_point = next(
                event.name for event in events
                if event.state is EventState.failure
            )
        except StopIteration:
            # A plausible cause is that the simulator timed out.
            # Then the probable failure point would probably be
            # one of the listening events.
            failure_point = next(
                event.name for event in events
                if event.state is EventState.awake
            )
            # If it fails though, that's definitely a bug.
        return False, failure_point


@memory.cache
def sample_valid_candidates(n_cand=2000):
    candidates = TeapotAdventure.sample_valid(n_cand, max_trials=10*n_cand,
                                              rule='R')
    print("Number of candidates:", len(candidates))
    return candidates


@memory.cache
def evaluate_random_candidates(n_cand=2000):
    samples = sample_valid_candidates(n_cand)
    results = Parallel(n_jobs=6)(delayed(run_and_check)(c) for c in samples)
    success_states, failure_points = zip(*results)
    print("Number of successful candidates:", sum(success_states))
    return samples, success_states, failure_points


def search_most_robust_solution(estimator: ScenarioRobustnessEstimator):
    candidates = TeapotAdventure.sample_valid(1000, max_trials=3000, rule='R')
    robustnesses = estimator.eval(candidates)
    return candidates[np.argmax(robustnesses)]


def view_solution(x, interactive=True):
    if interactive:
        scenario = TeapotAdventure(x, make_geom=True, graph_view=True)
        simu = Simulation(scenario, timestep=1/500)
        simu.run_visual(grid='xz', view_h=180)
    else:
        export_animation(x, filename="data/scene")
        replay_solution("data/scene")


def export_animation(x, filename="scene"):
    scenario = TeapotAdventure(x, make_geom=True)
    observer = StateObserver(scenario._scene)

    scenario.export_scene_to_egg(filename)
    simu = Simulation(scenario, [observer], timestep=1/500)
    simu.run()
    observer.export(filename+"_frames.pkl")


def replay_solution(name):
    app = Replayer(name+".bam", name+"_frames.pkl", grid='xz', view_h=180)
    try:
        app.run()
    except SystemExit:
        app.destroy()


def export(x, name):
    scenario = TeapotAdventure(x, make_geom=True)
    scenario.export_scene_to_pdf(name, x, (3*21, 2*29.7))


def show_failure_point_histogram(failure_points):
    all_events = [
        event.name for event in TeapotAdventure(
            cfg.MANUAL_SCENARIO_PARAMETERS
        ).causal_graph.get_events()
    ]
    cnt = Counter(failure_points)
    all_events.sort()
    counts = np.zeros(len(all_events))
    for i, event in enumerate(all_events):
        try:
            counts[i] = cnt[event]
        except KeyError:
            pass
    fig, ax = plt.subplots()
    width = 0.3
    x = np.arange(1, len(all_events) + 1)
    ax.bar(x, counts, width=width)
    ax.set_xticks(x)
    ax.set_xticklabels(all_events, rotation=45,
                       rotation_mode='anchor', ha='right')
    fig.subplots_adjust(bottom=.3)
    ax.set_title("Number of failures for each event "
                 "($N_s={}$)".format(len(failure_points)))
    plt.show()


def main():
    x_manual = cfg.MANUAL_SCENARIO_PARAMETERS
    # view_solution(x_manual)
    bounds = np.sort(np.column_stack((x_manual, x_manual))*[.95, 1.05], axis=1)
    bounds[-2] = [-.01, .01]
    cfg.SCENARIO_PARAMETERS_BOUNDS = bounds

    samples, success_states, failure_points = evaluate_random_candidates()
    # for x_random, success in zip(samples, success_states):
    #     if success:
    #         view_solution(x_random)
    show_failure_point_histogram(failure_points)
    # filename = "full-robustness.pkl"
    # try:
    #     full_rob_estimator = load(filename)
    # except FileNotFoundError:
    #     full_rob_estimator = ScenarioRobustnessEstimator(TeapotAdventure)
    #     full_rob_estimator.train(n_samples=2000, verbose=True)
    #     dump(full_rob_estimator, filename)
    # x_full_rob = search_most_robust_solution(full_rob_estimator)
    # view_solution(x_full_rob)
    # export(x_full_rob, filename[:-3] + "pdf")


if __name__ == "__main__":
    main()
