import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from joblib import Memory, Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                       SelectPercentile, mutual_info_classif)
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath("../.."))
import xp.adventure.config as cfg  # noqa: E402
from gui.viewers import Replayer  # noqa: E402
from xp.adventure.scenarios import StateObserver, TeapotAdventure  # noqa: E402
from xp.causal import EventState  # noqa: E402
from xp.robustness import EventRobustnessEstimator  # noqa: E402
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


def sample_valid_around(x, n_samples, radius):
    x = np.asarray(x).reshape(-1, 1)
    intervals = np.array(cfg.SCENARIO_PARAMETERS_INTERVALS)
    bounds = intervals * radius + x
    distribution = TeapotAdventure.build_distribution_from_bounds(bounds)
    samples = TeapotAdventure.sample_valid(
        n_samples, 30*n_samples, 'R', distribution
    )
    return samples


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


def smooth_min(x, alpha=cfg.SOFTMIN_COEFF):
    exps = np.exp(-alpha*x)
    return (x*exps).sum() / exps.sum()


@memory.cache
def search_most_robust_solution(estimators, weights):
    init = cfg.MANUAL_SCENARIO_PARAMETERS
    bounds = cfg.SCENARIO_PARAMETERS_BOUNDS
    constraints = [
        {'type': 'ineq',
         'fun': TeapotAdventure.get_physical_validity_constraint
         },
    ]
    events, estimators = zip(*estimators.items())
    weights = np.array([weights[e] for e in events])

    def objective(x):
        return -smooth_min(weights * [e.eval([x]) for e in estimators])

    solution = opt.minimize(
        objective, init,
        bounds=bounds, constraints=constraints,
        options=dict(disp=True)
    ).x
    return solution


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


def compute_event_assignment(samples, event, states, selector):
    event_samples = []
    event_values = []
    for state, sample in zip(states, samples):
        if state is EventState.failure:
            event_samples.append(sample)
            event_values.append(True)
        elif state is EventState.success:
            event_samples.append(sample)
            event_values.append(False)
    print("Computing parameter sensitivity for "
          "event {} with {} samples".format(event, len(event_samples)))
    selector.fit(event_samples, event_values)
    return selector.get_support(indices=True)


@memory.cache
def compute_assignment(samples, results, selector):
    if selector == 'kbest':
        selector = SelectKBest(mutual_info_classif, k=4)
    elif selector == 'percentile':
        selector = SelectPercentile(mutual_info_classif, percentile=10)
    elif selector == 'rfe':
        estimator = SVC(kernel='linear', C=1)
        selector = RFE(estimator, n_features_to_select=4, step=1)
    elif selector == 'model_svc':
        estimator = SVC(kernel='linear', C=1)
        selector = SelectFromModel(estimator, threshold="2*mean")
    elif selector == 'model_trees':
        estimator = ExtraTreesClassifier()
        selector = SelectFromModel(estimator, threshold="1.1*mean")
    results = list(results.items())
    assignment = Parallel(n_jobs=cfg.NCORES)(
        delayed(compute_event_assignment)(samples, event, states, selector)
        for event, states in results
    )
    return {e: a for (e, _), a in zip(results, assignment)}


@memory.cache
def train_robustness_estimators(assignment, n_samples=100):
    x_default = np.asarray(cfg.MANUAL_SCENARIO_PARAMETERS)
    estimators = {}
    for event, ids in assignment.items():
        print("\nTraining for event {}".format(event))
        param_filter = x_default.copy()
        param_filter[ids] = np.nan
        estimator = EventRobustnessEstimator(
            TeapotAdventure, event, param_filter
        )
        estimator.train(n_samples, verbose=True)
        estimators[event] = estimator
    return estimators


@memory.cache
def compute_success_rate_wrt_error(x, n_err=10,  n_samples=100):
    success_rates = np.zeros(n_err)
    for i, err in enumerate(np.linspace(0, 1, n_err+1)[1:]):
        samples = sample_valid_around(x, n_samples, err)
        results = Parallel(n_jobs=cfg.NCORES)(
            delayed(run_and_check_global_success)(s) for s in samples
        )
        success_rates[i] = sum(results) / len(results)
    return success_rates


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


def show_assigment(assignment, selector):
    assignment = list(assignment.items())
    assignment.sort()
    names, ind_lists = zip(*assignment)
    ass_matrix = np.zeros((len(names), len(cfg.PARAMETER_LABELS)), dtype=bool)
    for ind, row in zip(ind_lists, ass_matrix):
        row[ind] = True
    fig, ax = plt.subplots()
    ax.matshow(ass_matrix.T, aspect='auto', cmap=plt.cm.Blues)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, rotation_mode='anchor', ha='left')
    ax.set_yticks(range(len(cfg.PARAMETER_LABELS)))
    ax.set_yticklabels(cfg.PARAMETER_LABELS)
    ax.set_title("Parameter assignment with {}".format(selector), pad=100)
    fig.subplots_adjust(left=.25, top=.8, bottom=.05)


def show_success_rates_wrt_error(success_rates):
    fig, ax = plt.subplots()
    for label, sr in success_rates.items():
        err = np.linspace(0, 1, len(sr)+1)[1:]
        ax.plot(err, sr, label=label)
    ax.legend()
    ax.set_title("Success_rates wrt error")


def show_failure_wrt_2_params(samples_xy, failure_points, event):
    is_fp = np.asarray(failure_points) == event
    x, y = np.asarray(samples_xy).T

    fig, ax = plt.subplots()
    ax.scatter(x[is_fp], y[is_fp], label="Fails", alpha=.5)
    ax.scatter(x[~is_fp], y[~is_fp], label="Failsn't", alpha=.5)
    ax.legend()
    ax.set_title("Failure at {}".format(event))


def main():
    x_manual = cfg.MANUAL_SCENARIO_PARAMETERS
    # view_solution(x_manual, 0)
    success_rates = {}
    success_rates['manual'] = compute_success_rate_wrt_error(x_manual,
                                                             n_samples=200)

    random_samples, results = evaluate_random_candidates(5003)
    # for x_random, res in zip(random_samples, results['end']):
    #     if res is EventState.success:
    #         view_solution(x_random)
    frequencies = compute_failure_frequencies(results)
    show_failure_point_histogram(frequencies, len(samples))
    filtered_results = {e: s for e, s in results.items()
                        if frequencies[e] > .01}

    selector = 'model_svc'
    assignment = compute_assignment(samples, filtered_results, selector)
    show_assigment(assignment, selector)
    plt.show()

    estimators = train_robustness_estimators(assignment, n_samples=2000)
    x_best = search_most_robust_solution(estimators, frequencies)
    success_rates['best'] = compute_success_rate_wrt_error(x_best,
                                                           n_samples=200)
    show_success_rates_wrt_error(success_rates)
    plt.show()
    # view_solution(x_best, interactive=False)
    # export(x_full_rob, filename[:-3] + "pdf")


if __name__ == "__main__":
    main()
