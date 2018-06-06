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
from xp.robustness import RobustnessEstimator  # noqa: E402
from xp.simulate import Simulation  # noqa: E402

memory = Memory(cachedir=".cache")


def run_and_check_global_success(x):
    instance = TeapotAdventure(x)
    simu = Simulation(instance)
    simu.run()
    return instance.succeeded()


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
    candidates = TeapotAdventure.sample_valid(n_cand, max_trials=30*n_cand,
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


def filter_samples(raw_samples, states, frequencies, threshold=.05):
    samples_out = {}
    states_out = {}
    for event, event_states in states.items():
        if frequencies[event] < threshold:
            continue
        success_samples = []
        failure_samples = []
        for x, s in zip(raw_samples, event_states):
            if s is EventState.success:
                success_samples.append(x)
            elif s is EventState.failure:
                failure_samples.append(x)
        samples_out[event] = success_samples + failure_samples
        states_out[event] = (
            [True] * len(success_samples) + [False] * len(failure_samples)
        )
    return samples_out, states_out


def sample_event_boundary(event, seeds, states,
                          n_samples=100, radius=.1, n_growth=10,
                          n_jobs=cfg.NCORES):
    print("Sampling the boundary of event {}".format(event))
    success_seeds = [seed for seed, state in zip(seeds, states) if state]
    failure_seeds = [seed for seed, state in zip(seeds, states) if not state]
    # Initialize both success and failure lists
    success_samples = []
    failure_samples = []
    done = False
    i = 0
    while not done:
        # Choose seed list from least populated list, to help keeping the
        # growth uniform.
        if len(success_samples) <= len(failure_samples):
            seed_list = success_seeds
        else:
            seed_list = failure_seeds
        # Check that there remains enough seeds.
        if len(seed_list) < n_growth:
            print("Ran out of seeds for event {}".format(event))
            break
        # Sample the seeds.
        # Use choice and not randint because we want unique ids.
        seeds_ids = np.random.choice(
            len(seed_list), replace=False, size=n_growth
        )
        seeds = [seed_list[sid] for sid in seeds_ids]
        # Start the growing.
        center = np.mean(seeds, axis=0)
        cands = []
        succ_cands = []
        fail_cands = []
        dists = np.zeros(n_growth)
        p_num = np.zeros(n_growth)
        p_mix = np.zeros(n_growth)
        for j, seed in enumerate(seeds):
            print("Evaluating seed group {} for {}".format(j, event))
            # Distance to the center of mass
            diff = center - seed
            dists[j] = diff.dot(diff)
            # Sample and evaluate candidates
            cands.append(sample_valid_around(seed, n_growth**2, radius))
            succ_cands.append([])
            fail_cands.append([])
            cands_states = Parallel(n_jobs=n_jobs)(
                delayed(run_and_check)(cand) for cand in cands[-1]
            )
            for cand, cand_states in zip(cands[-1], cands_states):
                state = cand_states[event]
                if state is EventState.success:
                    succ_cands[-1].append(cand)
                elif state is EventState.failure:
                    fail_cands[-1].append(cand)
                elif state is EventState.asleep and event == "end":
                    fail_cands[-1].append(cand)
            # Compute probabilities
            n_succ = len(succ_cands[-1])
            n_fail = len(fail_cands[-1])
            p_num[j] = (n_succ + n_fail) / len(cands[-1])
            if n_succ or n_fail:
                p_mix[j] = min(n_succ, n_fail) / max(n_succ, n_fail)
        p_far = dists / dists.max()
        p_total = p_num * p_mix * p_far
        p_total_sum = p_total.sum()
        print("p_num", p_num)
        print("p_mix", p_mix)
        print("p_far", p_far)
        if p_total_sum:
            p_total /= p_total_sum
            # Pick seed from probability distribution
            seed_id = np.random.choice(n_growth, p=p_total)
            # Update lists
            del seed_list[seeds_ids[seed_id]]
            success_seeds += succ_cands[seed_id]
            failure_seeds += fail_cands[seed_id]
            success_samples += succ_cands[seed_id]
            failure_samples += fail_cands[seed_id]
        else:
            # All the seeds in this batch are bad
            for seed_id in sorted(seeds_ids, reverse=True):
                del seed_list[seed_id]
        # Check break condition
        print("{} successes and {} failures at iteration {} for {}".format(
            len(success_samples), len(failure_samples), i, event
        ))
        if (len(success_samples) >= n_samples
                and len(failure_samples) >= n_samples):
            break
        i += 1
    samples_out = success_samples + failure_samples
    states_out = [True] * len(success_samples) + [False] * len(failure_samples)
    return samples_out, states_out


# Don't use a decorator here because we are parallelizing it.
cached_sample_event_boundary = memory.cache(sample_event_boundary)


@memory.cache
def sample_boundaries(seeds, states, n_samples=100, radius=.1, n_growth=10):
    events, events_states = zip(*states.items())
    results = Parallel(n_jobs=cfg.NCORES)(
        delayed(cached_sample_event_boundary)(
            e, seeds[e], es, n_samples, radius, n_growth
        )
        for e, es in zip(events, events_states)
    )
    samples_out = {e: r[0] for e, r in zip(events, results)}
    states_out = {e: r[1] for e, r in zip(events, results)}
    return samples_out, states_out


@memory.cache
def evaluate_random_candidates(n_cand=100):
    samples = sample_valid_candidates(n_cand)
    results = Parallel(n_jobs=cfg.NCORES)(
        delayed(run_and_check)(c) for c in samples
    )
    # Aggregate results.
    results_agg = {e: [r[e] for r in results] for e in results[0].keys()}
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
    print("Computing parameter sensitivity for "
          "event {} with {} samples".format(event, len(samples)))
    selector.fit(samples, states)
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
        selector = SelectFromModel(estimator, threshold="1.2*mean")
    results = list(results.items())
    assignment = Parallel(n_jobs=cfg.NCORES)(
        delayed(compute_event_assignment)(samples[e], e, s, selector)
        for e, s in results
    )
    return {e: a for (e, _), a in zip(results, assignment)}


@memory.cache
def train_robustness_estimators(samples, states, assignment):
    estimators = {}
    for event, ids in assignment.items():
        print("\nTraining for event {}".format(event))
        estimator = RobustnessEstimator(TeapotAdventure, ids)
        estimator.train(samples[event], states[event], verbose=True)
        estimators[event] = estimator
    return estimators


@memory.cache
def train_full_robustness_estimator(samples, states):
    estimator = RobustnessEstimator(TeapotAdventure)
    estimator.train(samples, states, verbose=True)
    return estimator


@memory.cache
def compute_success_rate_wrt_error(x, n_err=10,  n_samples=100):
    success_rates = np.zeros(n_err)
    errors = np.logspace(-3, 0, n_err)
    for i, err in enumerate(errors):
        samples = sample_valid_around(x, n_samples, err)
        results = Parallel(n_jobs=cfg.NCORES)(
            delayed(run_and_check_global_success)(s) for s in samples
        )
        success_rates[i] = sum(results) / len(results)
    return errors, success_rates


def compute_success_rate_wrt_error_from_samples(x, samples, states, n_err=10):
    success_rates = np.zeros(n_err)
    max_vals = np.array(cfg.SCENARIO_PARAMETERS_INTERVALS)[:, 1]
    states = np.array(states)
    diffs = np.abs(np.array(samples) - np.array(x))
    errors = np.logspace(-6, 0, n_err)
    for i, err in enumerate(errors):
        print(diffs.shape, max_vals.shape)
        select = np.all(diffs < (err * max_vals), axis=1)
        local_states = states[select]
        success_rates[i] = local_states.sum() / local_states.size
    print(success_rates)
    return errors, success_rates


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
    scenario.export_scene_to_pdf(name, x, (5*21, 3*29.7))


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


def show_n_successful_events_histogram(n_successful_events):
    fig, ax = plt.subplots()
    ax.hist(n_successful_events)
    ax.set_title("Histogram of the nb of successful events for each sample")


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


def show_success_rate_decays(success_rate_decays):
    fig, ax = plt.subplots()
    for name, srd in success_rate_decays.items():
        success_rate_decays[name] = (
            [0] + list(srd[0][10:]),
            [1] + list(srd[1][10:])
        )
    ax.plot(*success_rate_decays['manual'], label="Expert")
    random_decays = np.array([sr[1] for name, sr in success_rate_decays.items()
                              if name.startswith("random")])
    avg_random_decay = random_decays.mean(axis=0)
    std_random_decay = random_decays.std(axis=0)
    errors = success_rate_decays["random_0"][0]
    n_random = random_decays.shape[0]
    ax.plot(errors, avg_random_decay, label="Random (n={})".format(n_random))
    ax.fill_between(errors,
                    avg_random_decay - std_random_decay,
                    avg_random_decay + std_random_decay,
                    color='tab:green', alpha=.5)
    ax.plot(*success_rate_decays['best'], label="Best")
    ax.legend()
    fig.tight_layout()
    # ax.set_title("Success rates wrt error (with spread for random)")


def show_failure_wrt_2_params(samples_xy, failure_points, event):
    is_fp = np.asarray(failure_points) == event
    x, y = np.asarray(samples_xy).T

    fig, ax = plt.subplots()
    ax.scatter(x[is_fp], y[is_fp], label="Fails", alpha=.5)
    ax.scatter(x[~is_fp], y[~is_fp], label="Failsn't", alpha=.5)
    ax.legend()
    ax.set_title("Failure at {}".format(event))


@memory.cache
def find_successful_samples_around(x0, n_samples=10, radius=.1, n_growth=10):
    x = x0
    seeds = []
    n_successful_events = []
    success_samples = []
    failure_samples = []
    while len(success_samples) < n_samples:
        # Compute new seeds
        new_samples = sample_valid_around(x, n_growth, radius).tolist()
        new_states = Parallel(n_jobs=cfg.NCORES)(
            delayed(run_and_check)(s) for s in new_samples
        )
        seeds += new_samples
        new_samples_nse = [
            sum(state is EventState.success for state in states.values())
            for states in new_states
        ]
        n_successful_events += new_samples_nse
        # Distribute between success and failure lists
        n_events = len(new_states[0])
        for new_sample, new_sample_nse in zip(new_samples, new_samples_nse):
            if new_sample_nse == n_events:
                success_samples.append(new_sample)
            else:
                failure_samples.append(new_sample)
        # Randomly pick  a new seed
        nse_sum = sum(n_successful_events)
        probas = [nse / nse_sum for nse in n_successful_events]
        new_id = np.random.choice(len(seeds), p=probas)
        x = seeds.pop(new_id)
        del n_successful_events[new_id]
        print("Number of successful samples so far:", len(success_samples))
    return success_samples, failure_samples


def main():
    # Dict of significant successes for comparison
    successes = {}
    # First evaluate random samples.
    random_samples, states = evaluate_random_candidates(3000)
    random_successes = [x for x, s in zip(random_samples, states["end"])
                        if s is EventState.success]
    print("Number of successes out of {} random samples: {}".format(
        len(random_samples), len(random_successes)
    ))
    for i, x in enumerate(random_successes):
        successes['random_{}'.format(i)] = x
    successes['manual'] = cfg.MANUAL_SCENARIO_PARAMETERS
    view_solution(successes['manual'])

    # Find best
    success_samples, failure_samples = find_successful_samples_around(
        successes['manual'], n_samples=30, radius=.15
    )
    print("Initial number of failures:", len(failure_samples))
    growth_samples = success_samples + failure_samples
    growth_states = ([True] * len(success_samples)
                     + [False] * len(failure_samples))
    bound_samples, bound_states = cached_sample_event_boundary(
        "end", growth_samples, growth_states,
        n_samples=150, radius=.2, n_growth=5
    )
    full_samples = growth_samples + bound_samples
    full_states = growth_states + bound_states
    estimator = train_full_robustness_estimator(full_samples, full_states)
    best_id = np.argmax(estimator.eval(full_samples))
    successes['best'] = full_samples[best_id]
    # Compute all success rate decays.
    success_rate_decays = {
        name: compute_success_rate_wrt_error(x, n_err=20, n_samples=200)
        for name, x in successes.items()
    }
    show_success_rate_decays(success_rate_decays)
    plt.show()
    view_solution(successes['best'], 1)
    # export(successes['best'], "data/best.pdf")
    return

    n_events = len(states)
    n_samples = len(random_samples)
    n_successful_events = np.zeros(n_samples)
    for event_states in states.values():
        n_successful_events += [s is EventState.success for s in event_states]
    n_full_success = (n_successful_events == n_events).sum()
    print("Initial number of successful candidates:", n_full_success)

    return
    # for x_random, res in zip(random_samples, states['end']):
    #     if res is EventState.success:
    #         view_solution(x_random, 0)
    frequencies = compute_failure_frequencies(states)
    show_failure_point_histogram(frequencies, len(random_samples))

    filtered_samples, filtered_states = filter_samples(
        random_samples, states, frequencies, threshold=.01
    )
    print("{} events to process: {}".format(
        len(filtered_samples), list(filtered_samples.keys())
    ))

    bound_samples, bound_states = sample_boundaries(
        filtered_samples, filtered_states,
        n_samples=30, radius=.1, n_growth=10
    )
    print("Samples computed around the boundary:",
          {e: len(s) for e, s in bound_samples.items()})

    full_samples = {
        e: filtered_samples[e] + bound_samples[e]
        for e in filtered_samples.keys()
    }
    full_states = {
        e: filtered_states[e] + bound_states[e]
        for e in filtered_states.keys()
    }
    selector = 'model_trees'
    assignment = compute_assignment(full_samples, full_states, selector)
    show_assigment(assignment, selector)
    plt.show()
    return

    estimators = train_robustness_estimators(
        full_samples, full_states, assignment
    )
    x_best = search_most_robust_solution(estimators, frequencies)
    # success_rates['best'] = compute_success_rate_wrt_error(x_best,
    #                                                        n_samples=200)
    # show_success_rates_wrt_error(success_rates)
    plt.show()
    view_solution(x_best, interactive=False)
    # export(x_full_rob, filename[:-3] + "pdf")


if __name__ == "__main__":
    main()
