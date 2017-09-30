"""
Time evaluation methods.

In the current version, each method returns the time at which each domino
topples.

"""
from functools import partial
from glob import glob
import os
import sys
#  import tempfile

import numpy as np
from sklearn.externals import joblib

sys.path.insert(0, os.path.abspath('..'))
from .config import w
from .config import TIMESTEP, TIMESTEP_PRECISE, MAX_WAIT_TIME
from .config import X_MAX, Y_MAX, A_MAX, MAX_SPACING
from domino_design.evaluate import setup_dominoes, get_toppling_angle

sys.path.insert(0, os.path.abspath('../..'))
import spline2d as spl


TIME_ESTIMATOR_PATHS = glob("data/latest/samples-4D-times-*-estimator.pkl")
TIME_ESTIMATOR_PATHS.sort()
ESTIMATORS = [joblib.load(path) for path in TIME_ESTIMATOR_PATHS]


def get_methods():
    physics_based_precise = partial(physics_based, ts=TIMESTEP_PRECISE)
    prev0_estimator = partial(nprev_estimator, nprev=0)
    prev1_estimator = partial(nprev_estimator, nprev=1)
    prev3_estimator = partial(nprev_estimator, nprev=3)
    prev4_estimator = partial(nprev_estimator, nprev=4)
    prev5_estimator = partial(nprev_estimator, nprev=5)
    prev6_estimator = partial(nprev_estimator, nprev=6)
    prev7_estimator = partial(nprev_estimator, nprev=7)
    return (physics_based_precise,
            physics_based,
            prev0_estimator,
            prev1_estimator,
            prev4_estimator,
            prev3_estimator,
            prev5_estimator,
            prev6_estimator,
            prev7_estimator,
            combined_estimators)



def _get_simu_top_times(u, spline, ts):
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
        time += ts
        world.do_physics(ts, 2, ts)
    return toppling_times


def physics_based(u, spline, ts=TIMESTEP):
    return _get_simu_top_times(u, spline, ts)


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
        return np.concatenate(([0.], np.cumsum(estimator.predict(xya))))
    else:
        length = np.asscalar(spl.arclength(spline, u[-1]))
        s = (length / len(u)) / MAX_SPACING
        xyas = np.hstack([xya, np.full((len(u)-1, 1), s)])
        return np.concatenate(([0.], np.cumsum(estimator.predict(xyas))))


def combined_estimators(u, spline):
    XYA = _get_relative_coords(u, spline)
    distances = XYA[:, 0] / MAX_SPACING
    xya = XYA / (X_MAX, Y_MAX, A_MAX)

    rel_times = np.empty(len(xya))

    rel_times[0] = np.asscalar(ESTIMATORS[0].predict([xya[0]]))

    x, y, a = xya[1]
    s = distances[0]
    rel_times[1] = np.asscalar(ESTIMATORS[1].predict([[x, y, a, s]]))

    x, y, a = xya[2]
    s = distances[:2].min()
    rel_times[2] = np.asscalar(ESTIMATORS[2].predict([[x, y, a, s]]))

    x, y, a = xya[3]
    s = distances[:3].min()
    rel_times[3] = np.asscalar(ESTIMATORS[3].predict([[x, y, a, s]]))

    x, y, a = xya[4]
    s = distances[:4].min()
    rel_times[4] = np.asscalar(ESTIMATORS[4].predict([[x, y, a, s]]))

    x, y, a = xya[5]
    s = distances[:5].min()
    rel_times[5] = np.asscalar(ESTIMATORS[5].predict([[x, y, a, s]]))

    s = np.array([[distances[i-6:i].min()] for i in range(6, len(xya))])
    rel_times[6:] = ESTIMATORS[6].predict(np.hstack((xya[6:], s)))

    return np.concatenate(([0.], np.cumsum(rel_times)))
