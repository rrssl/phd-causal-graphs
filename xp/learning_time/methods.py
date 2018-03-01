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

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl
from xp.config import X_MAX, Y_MAX, A_MAX, MAX_SPACING
from xp.dominoes.dompath import get_rel_coords
from xp.predicting_domino_timing.config import TIMESTEP_FAST, TIMESTEP_PRECISE
import xp.simulate as simu


TIME_ESTIMATOR_PATHS = glob(
        os.path.dirname(__file__)
        + "/data/latest/samples-4D-times-*-estimator.pkl")
TIME_ESTIMATOR_PATHS.sort()
ESTIMATORS = [joblib.load(path) for path in TIME_ESTIMATOR_PATHS]


def get_methods():
    physics_based_precise = partial(physics_based, ts=TIMESTEP_PRECISE)
    nprev_estimators = [partial(nprev_estimator, nprev=i)
                        for i in range(len(ESTIMATORS))]
    return [physics_based_precise, physics_based] + nprev_estimators + [
            combined_estimators]


def physics_based(u, spline, ts=TIMESTEP_FAST):
    doms_np, world = simu.setup_dominoes_from_path(u, spline)
    return simu.run_simu(doms_np, world, ts)


def nprev_estimator(u, spline, nprev):
    estimator = ESTIMATORS[nprev]
    xya = get_rel_coords(u, spline) / (X_MAX, Y_MAX, A_MAX)

    assert nprev >= 0
    if nprev == 0:
        return np.concatenate(([0.], np.cumsum(estimator.predict(xya))))
    else:
        length = np.asscalar(spl.arclength(spline, u[-1]))
        s = (length / len(u)) / MAX_SPACING
        xyas = np.hstack([xya, np.full((len(u)-1, 1), s)])
        return np.concatenate(([0.], np.cumsum(estimator.predict(xyas))))


def combined_estimators(u, spline):
    XYA = get_rel_coords(u, spline)
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
