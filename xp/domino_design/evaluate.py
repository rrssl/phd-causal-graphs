"""
Compute all the evaluation criteria for the input domino distributions.

For each distribution, the criteria are:
1. bool: the path is fully covered.
2. bool: the distribution is physically possible (no overlap between
    dominoes).
3. bool: the physically feasible distribution(*) topples entirely during
    simulation.
4. float in [0,1]: how many dominoes toppled during the previous simulation,
    as a fraction of the length thus covered over the total path length.
5. float in [0,1]: same as (4), but averaged over N=10 trials with a
    5% randomization(**) of the dominoes coordinates.
6-7. float in [0,1]: same as (5), with an uncertainty of respectively
    10 and 15%.

(*): if a distribution is physically impossible (i.e. criterion (2) is False),
it is made possible by randomly removing offending dominoes until there is no
overlap.
(**): for each coordinate, the uncertainty is taken as a percentage of the
'reasonable' range manually defined for all the experiments.

Parameters
----------
splpath : string
  Path to the list of splines
dompath : string
  Path to the domino runs.
ns : int, optional
  Only process the ns first splines of the list.

"""
import os
import pickle
import random
import sys

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import box
from sklearn.externals.joblib import Parallel, delayed

sys.path.insert(0, os.path.abspath("../.."))
import core.spline2d as spl  # noqa: E402
import xp.simulate as simu  # noqa: E402
from xp.config import NCORES, TOPPLING_ANGLE, h, t, w  # noqa: E402
from xp.domino_design.config import NTRIALS_UNCERTAINTY  # noqa: E402


VERBOSE = True


def test_path_coverage(u, spline):
    return bool((spl.arclength(spline) - spl.arclength(spline, u[-1])) < h)


def get_overlapping_dominoes(u, spline):
    """Get the indices of the dominoes that overlap."""
    overlapping = []
    if len(u) < 2:
        return overlapping
    base = box(-t/2, -w/2, t/2,  w/2)
    x, y = spl.splev(u, spline)
    a = spl.splang(u, spline)
    dominoes = [translate(rotate(base, ai), xi, yi)
                for xi, yi, ai in zip(x, y, a)]
    # Not efficient but who cares
    for i, d1 in enumerate(dominoes):
        for d2 in dominoes:
            if d2 is not d1 and d1.intersects(d2):
                overlapping.append(i)
                break

    return overlapping


def test_no_overlap(u, spline):
    return not bool(get_overlapping_dominoes(u, spline))


def get_successive_overlapping_dominoes(u, spline):
    """Get the indices of the successive dominoes that overlap."""
    overlapping = []
    if len(u) < 2:
        return overlapping
    base = box(-t/2, -w/2, t/2,  w/2)
    x, y = spl.splev(u, spline)
    a = spl.splang(u, spline)
    dominoes = [translate(rotate(base, ai), xi, yi)
                for xi, yi, ai in zip(x, y, a)]
    for d1, d2 in zip(dominoes[1:], dominoes[:-1]):
        if d1.intersects(d2):
            overlapping.append(d1)

    return overlapping


def test_no_successive_overlap(u, spline):
    return not bool(get_successive_overlapping_dominoes(u, spline))


def test_no_successive_overlap_fast(u, spline):
    """Like test_no_successive_overlap, but return as soon as one is found."""
    if len(u) < 2:
        return True
    base = box(-t/2, -w/2, t/2,  w/2)
    x, y = spl.splev(u, spline)
    a = spl.splang(u, spline)
    d1 = None
    d2 = translate(rotate(base, a[0]), x[0], y[0])
    for i in range(1, len(u)):
        d1 = d2
        d2 = translate(rotate(base, a[i]), x[i], y[i])
        if d1.intersects(d2):
            return False
    return True


def get_toppling_fraction(u, spline, doms_np):
    n = doms_np.get_num_children()
    try:
        # Find the first domino that didn't topple.
        idx = next(i for i in range(n)
                   if doms_np.get_child(i).get_r() < TOPPLING_ANGLE)
    except StopIteration:
        # All dominoes toppled
        idx = n
    return spl.arclength(spline, u[idx-1]) / spl.arclength(spline)


def test_all_toppled(doms_np):
    return all(dom.get_r() >= TOPPLING_ANGLE for dom in doms_np.get_children())


def evaluate_domino_run(u, spline, _id=None):
    """Evaluate the domino distribution for all criteria."""
    if VERBOSE:
        print(_id, ": Starting evaluation.")
        print(_id, ": Starting deterministic tests.")
    covered = test_path_coverage(u, spline)
    non_overlapping = test_no_successive_overlap_fast(u, spline)
    # If not valid, make a valid version
    u_valid = list(u)
    if not non_overlapping:
        overlap = get_overlapping_dominoes(u, spline)
        while overlap:
            u_valid.pop(random.choice(overlap))
            overlap = get_overlapping_dominoes(u_valid, spline)
    # Run simulation without uncertainty
    doms_np, world = simu.setup_dominoes_from_path(u_valid, spline)
    simu.run_simu(doms_np, world)
    all_topple = test_all_toppled(doms_np)
    top_frac = get_toppling_fraction(u_valid, spline, doms_np)

    deterministic_results = [covered, non_overlapping, all_topple, top_frac]
    if VERBOSE:
        print(_id, ": Starting randomized tests.")

    # Run simulations with uncertainty
    top_frac_rnd = []
    for uncertainty in (.05, .1, .15):
        if VERBOSE:
            print(_id, ": Testing with uncertainty = ", uncertainty)
        top_frac_rnd_trials = np.empty(NTRIALS_UNCERTAINTY)
        for i in range(NTRIALS_UNCERTAINTY):
            doms_np, world = simu.setup_dominoes_from_path(
                    u_valid, spline, uncertainty)
            simu.run_simu(doms_np, world)
            top_frac_rnd_trials[i] = get_toppling_fraction(
                    u_valid, spline, doms_np)
        top_frac_rnd.append(top_frac_rnd_trials.mean())

    if VERBOSE:
        print(_id, ": Done.")

    return deterministic_results + top_frac_rnd


def get_additional_metrics(u, spline):
    length = spl.arclength(spline, u[-1])
    density = (len(u) * t / length) if length else 1
    return [density]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    splpath = sys.argv[1]
    dompath = sys.argv[2]
    ns = int(sys.argv[3]) if len(sys.argv) == 4 else None

    with open(splpath, 'rb') as fs:
        splines = pickle.load(fs)[slice(ns)]
    domruns = np.load(dompath)

    results = Parallel(n_jobs=NCORES, verbose=VERBOSE)(
            delayed(evaluate_domino_run)(domruns['arr_{}'.format(i)], s, i)
            for i, s in enumerate(splines))
    #  results = [evaluate_domino_run(domruns['arr_{}'.format(i)], s, i)
    #             for i, s in enumerate(splines)]

    metrics = Parallel(n_jobs=NCORES)(
            delayed(get_additional_metrics)(domruns['arr_{}'.format(i)], s)
            for i, s in enumerate(splines))

    dirname = os.path.dirname(dompath)
    prefix = os.path.splitext(os.path.basename(dompath))[0]
    outname = prefix + "-validity.npy"
    np.save(os.path.join(dirname, outname), results)
    outname = prefix + "-metrics.npy"
    np.save(os.path.join(dirname, outname), metrics)


if __name__ == "__main__":
    main()
