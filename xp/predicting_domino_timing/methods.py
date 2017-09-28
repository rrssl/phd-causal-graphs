"""
Time evaluation methods.

In the current version, each method returns the total toppling time.
"""
from functools import partial
from glob import glob
import os
import sys
#  import tempfile

import numpy as np
from sklearn.externals import joblib

sys.path.insert(0, os.path.abspath('..'))
from .config import timestep, MAX_WAIT_TIME
from .config import X_MAX, Y_MAX, A_MAX, MAX_SPACING
from domino_design.evaluate import setup_dominoes, get_toppling_angle

sys.path.insert(0, os.path.abspath('../..'))
import spline2d as spl


TIME_ESTIMATOR_PATHS = glob("data/latest/samples-4D-times-*-estimator.pkl")
TIME_ESTIMATOR_PATHS.sort()
ESTIMATORS = [joblib.load(path) for path in TIME_ESTIMATOR_PATHS]


def get_methods():
    prev0_estimator = partial(nprev_estimator, nprev=0)
    prev1_estimator = partial(nprev_estimator, nprev=1)
    prev6_estimator = partial(nprev_estimator, nprev=6)
    return (physics_based,
            prev0_estimator,
            prev1_estimator,
            prev6_estimator,
            combined_estimators)



def _get_simu_top_times(u, spline):
    doms_np, world = setup_dominoes(u, spline)
    n = len(u)
    dominoes = list(doms_np.get_children())
    last_toppled_id = -1
    toppling_times = np.full(n, np.inf)
    time = 0.
    toppling_angle = get_toppling_angle()
    while True:
        if dominoes[last_toppled_id+1].get_r() >= toppling_angle:
            last_toppled_id += 1
            toppling_times[last_toppled_id] = time
        if last_toppled_id == n-1:
            # All dominoes toppled in order.
            break
        if time - toppling_times[last_toppled_id] > MAX_WAIT_TIME:
            # The chain broke
            break
        time += timestep
        world.do_physics(timestep, 2, timestep)
    return toppling_times


def physics_based(u, spline):
    return _get_simu_top_times(u, spline).max()


#  cachedir = tempfile.mkdtemp()
#  memory = joblib.Memory(cachedir=cachedir, verbose=0)
#  @memory.cache
def _get_relative_coords(u, spline):
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

    return np.column_stack((xi, yi, ai))


def nprev_estimator(u, spline, nprev):
    estimator = ESTIMATORS[nprev]
    xya = _get_relative_coords(u, spline) / (X_MAX, Y_MAX, A_MAX)

    assert nprev >= 0
    if nprev == 0:
        return estimator.predict(xya).sum()
    else:
        length = np.asscalar(spl.arclength(spline, u[-1]))
        s = (length / len(u)) / MAX_SPACING
        xyas = np.hstack([xya, np.full((len(u)-1, 1), s)])
        return estimator.predict(xyas).sum()


def combined_estimators(u, spline):
    XYA = _get_relative_coords(u, spline)
    distances = np.sqrt(XYA[:, 0]**2 + XYA[:, 1]**2)
    xya = XYA / (X_MAX, Y_MAX, A_MAX)

    rel_times = np.empty(len(xya))
    for i, (x, y, a) in enumerate(xya):
        nprev = min(i, len(ESTIMATORS)-1)
        estimator = ESTIMATORS[nprev]
        if i == 0:
            rel_times[i] = np.asscalar(estimator.predict([[x, y, a]]))
        else:
            s = distances[i-nprev:i].mean() / MAX_SPACING
            rel_times[i] = np.asscalar(estimator.predict([[x, y, a, s]]))
    return sum(rel_times)
