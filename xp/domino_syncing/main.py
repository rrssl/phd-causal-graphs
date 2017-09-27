"""
Define a method to constrain a chain to path extremities, as well as toppling
duration; run an experiment to evaluate this method.

Parameters
----------
spath : string
  Path to the .pkl file of splines.
sid : int
  Index of the spline in that file.

"""
import math
import os
import pickle
import sys
import tempfile

import numpy as np
import scipy.optimize as opt
from sklearn.externals import joblib

sys.path.insert(0, os.path.abspath('..'))
from domino_syncing.config import t, h
from domino_syncing.config import X_MAX, Y_MAX, A_MAX
from domino_design.methods import equal_spacing, batch_classif_based
from predicting_domino_timing.process_splines import compute_times

sys.path.insert(0, os.path.abspath('../..'))
import spline2d as spl


TIME_FRAC = .75
ALPHA = .1
TIME_ESTIMATOR_PATH = "../predicting_domino_timing/data/latest/samples-3D-estimator.pkl"
VALIDATOR_PATH = "../predicting_domino_toppling/data/latest/samples-3D-classifier.pkl"
estimator = joblib.load(TIME_ESTIMATOR_PATH)
validator = joblib.load(VALIDATOR_PATH)


cachedir = tempfile.mkdtemp()
memory = joblib.Memory(cachedir=cachedir, verbose=0)
@memory.cache
def param2coords(u, spline):
    # Get local Cartesian coordinates
    # Change origin
    xi, yi = spl.splev(u, spline)
    xi = xi[1:] - xi[:-1]
    yi = yi[1:] - yi[:-1]
    # Rotate by -a_i-1
    ai = spl.splang(u, spline, degrees=False)
    ci_ = np.cos(ai[:-1])
    si_ = np.sin(ai[:-1])
    xi = xi*ci_ + yi*si_
    yi = -xi*si_ + yi*ci_
    # Get relative angles
    ai = np.degrees(ai[1:] - ai[:-1])
    ai = (ai + 180) % 360 - 180  # Convert from [0, 360) to [-180, 180)
    # Symmetrize
    ai = np.copysign(ai, yi)
    yi = abs(yi)
    # Normalize
    xi /= X_MAX
    yi /= Y_MAX
    ai /= A_MAX
    return np.column_stack((xi, yi, ai))


def get_times(u, spline):
    return estimator.predict(param2coords(u, spline))


def get_validities(u, spline):
    return validator.predict(param2coords(u, spline))


def get_robustness(u, spline):
    return -validator.decision_function(param2coords(u, spline)).mean()


def init_optim(spline, time):
    # There's a limited number of valid domino run sizes so we just use brute
    # force.
    length = spl.arclength(spline)
    n_min = math.floor(length / h) + 1
    n_max = math.floor(length / t) - 1
    u_least_dense = equal_spacing(spline, n_min)
    time_least_dense = get_times(u_least_dense, spline).sum()
    u_most_dense = equal_spacing(spline, n_max)
    time_most_dense = get_times(u_most_dense, spline).sum()
    time_min = min(time_least_dense, time_most_dense)
    time_max = max(time_least_dense, time_most_dense)
    print("Min predicted reachable time: ", time_min)
    print("Max predicted reachable time: ", time_max)
    assert time_min < time < time_max, "Expected time cannot be reached"
    if abs(time - time_least_dense) < abs(time - time_most_dense):
        t_init = time_least_dense
        n_init = n_min
        u_init = u_least_dense
    else:
        t_init = time_most_dense
        n_init = n_max
        u_init = u_most_dense
    valid_distrib = False
    for ni in range(n_min+1, n_max):
        ui = equal_spacing(ni)
        ti = get_times(ui, spline)
        vi = eval_pairs_in_distrib(ui, spline, validator)
        if vi.all() and abs(time - t_init) < abs(time - ti):
            estimated_time = ti
            n_init = ni
            u_init = ui
            valid_distrib = True
    assert valid_distrib, "Could not find a valid initial distribution"
    print("Initial time: ", t_init)

    return n_init, u_init


def optimize_for_time(spline, time):
    # Initialization: determine the number of dominoes.
    n_init, u_init = init_optim(spline, time)
    # Global optimization

    def objective(u):
        return ((time - get_times(u, spline).sum())**2
                + ALPHA * get_robustness(u, spline))

    cons = ()  # TODO
    u_opt = opt.fmin_cobyla(objective, u_init, cons, rhobeg=1e-3,
                            disp=0)

    return u_opt


def run_xp(spline):
    # Populate the spline with dominoes.
    u_init = batch_classif_based(spline, batchsize=1)
    # Run a simulation to evaluate the toppling time.
    times = compute_times(u_init, spline)
    print(times[1:] - times[:-1])
    toppling_time = np.max(times)
    print("Initial toppling time: ", toppling_time)
    print("Initial predicted time: ", get_times(u_init, spline).sum())
    print(get_times(u_init, spline))
    assert toppling_time != np.inf, "Couldn't find valid distrib for input."
    # Run optimization to complete the run in a fraction of that time.
    expected_time = TIME_FRAC * toppling_time
    print("Expected time: ", expected_time)
    u_opt = optimize_for_time(spline, expected_time)
    # Run a new simulation to validate that new time.
    new_toppling_time = np.max(compute_times(u_opt, spline))

    return expected_time, new_toppling_time


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])
    with open(spath, 'rb') as f:
        spline = pickle.load(f)[sid]

    expected_time, toppling_time = run_xp(spline)
    print("Toppling time: {}; for an expected time of {}".format(
        toppling_time, expected_time))


if __name__ == "__main__":
    main()
