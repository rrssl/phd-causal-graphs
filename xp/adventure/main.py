import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory, dump, load, Parallel, delayed
from sklearn.feature_selection import mutual_info_classif

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
    causal_states = {
        e.name: e.state for e in scenario.causal_graph.get_events()
    }
    return causal_states


@memory.cache
def sample_valid_candidates(n_cand=100):
    candidates = TeapotAdventure.sample_valid(n_cand, max_trials=10*n_cand,
                                              rule='R')
    print("Number of candidates:", len(candidates))
    return candidates


@memory.cache
def evaluate_random_candidates(n_cand=100):
    samples = sample_valid_candidates(n_cand)
    results = Parallel(n_jobs=6)(delayed(run_and_check)(c) for c in samples)
    # Aggregate results.
    results_agg = {k: [r[k] for r in results] for k in results[0].keys()}
    n_success = sum(s is EventState.success for s in results_agg['end'])
    print("Number of successful candidates:", n_success)
    return samples, results_agg


def compute_failure_frequencies(results):
    events, events_states = zip(*results.items())
    n_samples = len(events_states[0])
    cnt = [sum(s is EventState.failure for s in es) for es in events_states]
    return {e: c / n_samples for e, c in zip(events, cnt)}


def search_most_robust_solution(estimator: ScenarioRobustnessEstimator):
    candidates = TeapotAdventure.sample_valid(1000, max_trials=3000, rule='R')
    robustnesses = estimator.eval(candidates)
    return candidates[np.argmax(robustnesses)]


@memory.cache
def compute_correlations(samples, results):
    scores = {}
    for event, states in results.items():
        corr_samples = []
        corr_values = []
        for state, sample in zip(states, samples):
            if state is EventState.failure:
                corr_samples.append(sample)
                corr_values.append(True)
            elif state is EventState.success:
                corr_samples.append(sample)
                corr_values.append(False)
        if sum(corr_values) < 2:
            # Not enough failures to get significant correlation scores
            continue
        mi = mutual_info_classif(corr_samples, corr_values)
        scores[event] = mi / mi.max()
    return scores


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


def show_failure_point_histogram(frequencies, n_samples):
    frequencies = list(frequencies.items())
    frequencies.sort()
    names, values = zip(*frequencies)
    fig, ax = plt.subplots()
    width = 0.3
    x = np.arange(1, len(names) + 1)
    ax.bar(x, values, width=width)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45,
                       rotation_mode='anchor', ha='right')
    fig.subplots_adjust(bottom=.3)
    ax.set_title("Failure frequency per event ($N_s={}$)".format(n_samples))


def show_correlation(scores):
    scores = list(scores.items())
    scores.sort()
    names, values = zip(*scores)
    values = np.array(values).T
    fig, ax = plt.subplots()
    im = ax.matshow(values, aspect='auto', cmap=plt.cm.Reds)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, rotation_mode='anchor', ha='left')
    ax.set_yticks(range(len(cfg.PARAMETER_LABELS)))
    ax.set_yticklabels(cfg.PARAMETER_LABELS)
    ax.set_title("Correlation between parameters and failure", pad=100)
    fig.subplots_adjust(left=.25, right=1, top=.8, bottom=.05)
    fig.colorbar(im)


def main():
    x_manual = cfg.MANUAL_SCENARIO_PARAMETERS
    # view_solution(x_manual)
    bounds = np.sort(np.column_stack((x_manual, x_manual))*[.95, 1.05], axis=1)
    bounds[-2] = [-.01, .01]
    cfg.SCENARIO_PARAMETERS_BOUNDS = bounds

    samples, results = evaluate_random_candidates(5000)
    # for x_random, res in zip(samples, results['end']):
    #     if res is EventState.success:
    #         view_solution(x_random)
    frequencies = compute_failure_frequencies(results)
    show_failure_point_histogram(frequencies, len(samples))
    filtered_results = {e: s for e, s in results.items()
                        if frequencies[e] > .01}

    scores = compute_correlations(samples, filtered_results)
    show_correlation(scores)
    plt.show()
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
