"""
Given two domino paths 1 and 2 with the same initial conditions, maximize the
difference of completion times T1 - T2.

Parameters
----------
spath1 : string
  path to the first .pkl spline file.
spath2 : string
  path to the second .pkl spline file.

"""
import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from shapely.affinity import rotate, translate
from sklearn.externals import joblib
import scipy.optimize as opt

from config import (MIN_SPACING, MAX_SPACING, NCORES, t, w)
from xp.dominoes.templates import create_line, create_branch
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl  # noqa
from xp.calibration.doms2pdf import export_domino_run  # noqa
from xp.dominoes.creation import equal_spacing  # noqa
from xp.domino_predictors import (DominoRobustness, DominoTime)  # noqa
from xp.simulate import Simulation  # noqa
from xp.viewdoms import DominoViewer  # noqa


ROB_COEFF = .3


def time_domino_runs(*runs_coords):
    simu = Simulation()
    for i, rc in enumerate(runs_coords):
        simu.add_domino_run(rc, tilt_first_dom=(i == 0))
    time = simu.run().max()
    return time


def get_linspace_coords(n, s):
    u = equal_spacing(s, n)
    x, y = spl.splev(u, s)
    a = spl.splang(u, s)
    return np.column_stack((x, y, a))


def run_bruteforce_optim(splines, bounds, bootstrap):
    s1, s2 = splines
    n1_rng = np.arange(bounds[0][0], bounds[0][1]+1)
    n2_rng = np.arange(bounds[1][0], bounds[1][1]+1)
    # Sample feasible domino distributions (equally spaced) and time them.
    times1 = joblib.Parallel(n_jobs=NCORES)(
            joblib.delayed(time_domino_runs)(
                bootstrap, get_linspace_coords(n, s1))
            for n in n1_rng)
    times2 = joblib.Parallel(n_jobs=NCORES)(
            joblib.delayed(time_domino_runs)(
                bootstrap, get_linspace_coords(n, s2))
            for n in n2_rng)
    # Find best combination.
    times = np.meshgrid(times1, times2, sparse=True)
    diff = times[1] - times[0]
    #  diff = np.abs(diff)  # if the objective is 'same time'
    diff[np.isneginf(diff)] = np.inf
    # /!\ The shape of meshgrid's output is the reverse of the inputs' shapes
    best_t2_id, best_t1_id = np.unravel_index(diff.argmin(), diff.shape)

    best_t1 = times1[best_t1_id]
    best_t2 = times2[best_t2_id]
    print("Best bruteforce time difference: ", best_t1 - best_t2)
    best_n1 = n1_rng[best_t1_id]
    best_n2 = n2_rng[best_t2_id]
    best_u1 = equal_spacing(s1, best_n1)
    best_u2 = equal_spacing(s2, best_n2)
    return best_u1, best_u2


class OptimModel:
    """Model used to map the optimizer vector to parameters usable by other
    functions.

    Some constraints are implicitly built into the model (e.g fixed dominoes).
    """
    def __init__(self, n1, n2, s1, s2):
        self.n1 = n1
        self.n2 = n2
        self.s1 = s1
        self.s2 = s2
        # Init c1
        c1 = np.zeros((n1, 3))
        c1[0, 0], c1[0, 1] = spl.splev(0, s1)
        c1[0, 2] = spl.splang(0, s1)
        c1[-1, 0], c1[-1, 1] = spl.splev(1, s1)
        c1[-1, 2] = spl.splang(1, s1)
        self.c1 = c1
        # Init c2
        c2 = np.zeros((n2, 3))
        c2[0, 0], c2[0, 1] = spl.splev(0, s2)
        c2[0, 2] = spl.splang(0, s2)
        c2[-1, 0], c2[-1, 1] = spl.splev(1, s2)
        c2[-1, 2] = spl.splang(1, s2)
        self.c2 = c2

        self._last_x = None

    def update(self, x):
        if np.array_equal(x, self._last_x):
            return
        n1_ = self.n1-2  # first and last domino are fixed
        n2_ = self.n2-2
        c1 = self.c1
        c2 = self.c2

        c1[1:-1, 0], c1[1:-1, 1] = spl.splev(x[:n1_], self.s1)
        c1[1:-1, 2] = x[n1_:2*n1_] * 360 - 180

        c2[1:-1, 0], c2[1:-1, 1] = spl.splev(x[2*n1_:2*n1_+n2_], self.s2)
        c2[1:-1, 2] = x[2*n1_+n2_:] * 360 - 180

        self._last_x = x.copy()

    @property
    def u1(self):
        n1_ = self.n1-2  # first and last domino are fixed
        return np.hstack((0, self._last_x[:n1_], 1))

    @property
    def u2(self):
        n1_ = self.n1-2  # first and last domino are fixed
        n2_ = self.n2-2
        return np.hstack((0, self._last_x[2*n1_:2*n1_+n2_], 1))


class Objective1:
    def __init__(self, model, rob_estimator, time_estimator):
        self.model = model
        self.rob_estimator = rob_estimator
        self.time_estimator = time_estimator

    def __call__(self, x):
        self.model.update(x)

        c1 = self.model.c1
        c2 = self.model.c2

        t1 = self.time_estimator(c1).sum()
        t2 = self.time_estimator(c2).sum()
        r1 = -self.rob_estimator(c1).mean()
        r2 = -self.rob_estimator(c2).mean()

        energy = (t2 - t1) + ROB_COEFF*(r1 + r2)
        return energy


class SequenceConstraint:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        self.model.update(x)
        return np.concatenate((np.diff(self.model.u1),
                               np.diff(self.model.u2)))


class NonPenetrationConstraint:
    def __init__(self, model):
        self.model = model
        self.base = box(-t/2, -w/2, t/2,  w/2)

    def _get_penetrations(self, coords):
        base = self.base
        b1 = [translate(rotate(base, a), x, y) for x, y, a in coords]
        return [-b1j.intersection(b1k).area / (t * w)
                for b1j, b1k in zip(b1[:-1], b1[1:])]

    def __call__(self, x):
        self.model.update(x)
        return np.concatenate((self._get_penetrations(self.model.c1),
                               self._get_penetrations(self.model.c2)))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath1 = sys.argv[1]
    spath2 = sys.argv[2]
    # Load paths
    with open(spath1, 'rb') as f1:
        s1 = pickle.load(f1)[0]
    with open(spath2, 'rb') as f2:
        s2 = pickle.load(f2)[0]

    # Arange paths on the plane (in a polished version this could be done
    # via a GUI)
    s1[1][0] *= -3.8
    s1[1][1] += .1
    s2[1][0] *= -4
    s2[1][1] *= 2.1
    # Create auxiliary domino runs.
    bootstrap = create_branch((.19, .05), 180, 12*t, 7*t, 1/(2*t))
    end1 = create_line((s1[1][0][-1]-12*t, s1[1][1][-1]), 180, 10*t, 1/(2*t))
    end2 = create_line((s2[1][0][-1], s2[1][1][-1]+12*t), 90, 10*t, 1/(2*t))
    switch = [[s1[1][0][-1]-6*t, s2[1][1][-1]+6*t, -45]]

    # Compute the search space.
    l1 = spl.arclength(s1)
    n1_min = math.ceil(l1 / MAX_SPACING)
    n1_max = math.floor(l1 / MIN_SPACING)
    l2 = spl.arclength(s2)
    n2_min = math.ceil(l2 / MAX_SPACING)
    n2_max = math.floor(l2 / MIN_SPACING)
    print("Length of path 1: {}, length of path 2: {}".format(l1, l2))

    # Show the result with naive spacing.
    n1 = int((n1_min+n1_max)/2)
    n2 = int((n2_min+n2_max)/2)
    u1_naive = equal_spacing(s1, n1)
    u2_naive = equal_spacing(s2, n2)
    d1_naive = np.column_stack(
            spl.splev(u1_naive, s1) + [spl.splang(u1_naive, s1)])
    d2_naive = np.column_stack(
            spl.splev(u2_naive, s2) + [spl.splang(u2_naive, s2)])
    t1_naive = time_domino_runs(bootstrap, d1_naive)
    t2_naive = time_domino_runs(bootstrap, d2_naive)
    print("Time difference with naive sampling: ", t1_naive - t2_naive)
    if 1:
        dom_viewer = DominoViewer()
        dom_viewer.add_domino_run(bootstrap)
        dom_viewer.add_domino_run(d1_naive, tilt_first_dom=False)
        dom_viewer.add_domino_run(d2_naive, tilt_first_dom=False)
        dom_viewer.add_domino_run(end1, tilt_first_dom=False)
        dom_viewer.add_domino_run(end2, tilt_first_dom=False)
        dom_viewer.add_domino_run(switch, tilt_first_dom=False)
        try:
            dom_viewer.run()
        except SystemExit:
            dom_viewer.destroy()
        #  return

    if 0:
        n_samples = 300
        i = 0
        time_diffs = np.zeros(n_samples)
        positive_times = []
        np.random.seed(1)
        x_bnd = .005
        a_bnd = 5
        while i < n_samples:
            x1_pert = np.random.uniform(-x_bnd, x_bnd, n1)
            y1_pert = np.random.uniform(-x_bnd, x_bnd, n1)
            a1_pert = np.random.uniform(-a_bnd, a_bnd, n1)
            x2_pert = np.random.uniform(-x_bnd, x_bnd, n2)
            y2_pert = np.random.uniform(-x_bnd, x_bnd, n2)
            a2_pert = np.random.uniform(-a_bnd, a_bnd, n2)
            d1 = d1_naive + np.column_stack((x1_pert, y1_pert, a1_pert))
            d2 = d2_naive + np.column_stack((x2_pert, y2_pert, a2_pert))
            t1 = time_domino_runs(bootstrap, d1)
            t2 = time_domino_runs(bootstrap, d2)
            if np.isfinite((t1, t2)).all():
                time_diffs[i] = t1 - t2
                i += 1
                if t1 > t2:
                    positive_times.append((t1-t2, d1, d2))
                if i % 10 == 0:
                    print(i, " done")
        plt.hist(time_diffs)
        plt.show()
        joblib.dump(positive_times, "positives_times.pkl")
        id_ = np.argmax([pt[0] for pt in positive_times])
        d1 = positive_times[id_][1]
        d2 = positive_times[id_][2]
        dom_viewer = DominoViewer()
        dom_viewer.add_domino_run(bootstrap)
        dom_viewer.add_domino_run(d1, tilt_first_dom=False)
        dom_viewer.add_domino_run(d2, tilt_first_dom=False)
        dom_viewer.add_domino_run(end1, tilt_first_dom=False)
        dom_viewer.add_domino_run(end2, tilt_first_dom=False)
        dom_viewer.add_domino_run(switch, tilt_first_dom=False)
        try:
            dom_viewer.run()
        except SystemExit:
            dom_viewer.destroy()
        return

    # Run an optimization to find best distribution.
    if 1:
        bounds = ((n1_min, n1_max), (n2_min, n2_max))
        u1_opt, u2_opt = run_bruteforce_optim((s1, s2), bounds, bootstrap)
        d1_opt = np.column_stack(
                spl.splev(u1_opt, s1) + [spl.splang(u1_opt, s1)])
        d2_opt = np.column_stack(
                spl.splev(u2_opt, s2) + [spl.splang(u2_opt, s2)])
    else:
        print("Starting optimization...")
        model = OptimModel(len(u1_naive), len(u2_naive), s1, s2)
        rob_estimator = DominoRobustness()
        time_estimator = DominoTime()
        objective = Objective1(model, rob_estimator, time_estimator)
        seq_cons = SequenceConstraint(model)
        nonpen_cons = NonPenetrationConstraint(model)
        # Define init
        x0 = np.concatenate((
                u1_naive[1:-1],
                (d1_naive[1:-1, 2] + 180) / 360,
                u2_naive[1:-1],
                (d2_naive[1:-1, 2] + 180) / 360,
                ))
        print(objective(x0))
        print(seq_cons(x0))
        print(nonpen_cons(x0))
        # Define additional options
        bounds = [(0., 1.)] * len(x0)
        cons = [
                {'type': 'ineq', 'fun': seq_cons},
                {'type': 'ineq', 'fun': nonpen_cons}]
        options = {'disp': True}
        # Run optim
        res = opt.minimize(objective, x0, bounds=bounds, constraints=cons,
                           options=options)
        print(res.message)
        print(objective(res.x))
        print(seq_cons(res.x))
        print(nonpen_cons(res.x))
        model.update(res.x)
        d1_opt = model.c1
        d2_opt = model.c2
        u1_opt, u2_opt = model.u1, model.u2
    t1_opt = time_domino_runs(bootstrap, d1_opt)
    t2_opt = time_domino_runs(bootstrap, d2_opt)
    print("Time difference with optimal sampling: ", t1_opt - t2_opt)
    # Visualize the result.
    dom_viewer = DominoViewer()
    dom_viewer.add_domino_run(bootstrap)
    dom_viewer.add_domino_run(d1_opt, tilt_first_dom=False)
    dom_viewer.add_domino_run(d2_opt, tilt_first_dom=False)
    dom_viewer.add_domino_run(end1, tilt_first_dom=False)
    dom_viewer.add_domino_run(end2, tilt_first_dom=False)
    dom_viewer.add_domino_run(switch, tilt_first_dom=False)
    try:
        dom_viewer.run()
    except SystemExit:
        dom_viewer.destroy()

    # Export the result for printing.
    naive_layout = np.vstack((bootstrap, d1_naive, d2_naive, end1, end2))
    best_layout = np.vstack((bootstrap, d1_opt, d2_opt, end1, end2))
    dirname = os.path.dirname(spath1)
    sheetsize = (2*21, 3*29.7)
    export_domino_run(os.path.join(dirname, "naive_layout"), naive_layout,
                      sheetsize)
    export_domino_run(os.path.join(dirname, "best_layout"), best_layout,
                      sheetsize)


if __name__ == "__main__":
    main()
