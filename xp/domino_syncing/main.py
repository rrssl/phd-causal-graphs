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
from functools import partial
import math
import os
import pickle
import sys

from interval import interval
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn.externals import joblib

from config import t, w
from config import X_MAX, Y_MAX, A_MAX, MIN_SPACING, MAX_SPACING
sys.path.insert(0, os.path.abspath(".."))
from domino_design.methods import equal_spacing, batch_classif_based
from xp.dominoes.dompath import get_rel_coords
from predicting_domino_timing.methods import physics_based
#  from predicting_domino_timing.methods import nprev_estimator
from predicting_domino_timing.methods import combined_estimators
from viewdoms import show_dominoes
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl
import export


TIME_FRAC = .75
ALPHA = .1
VALIDATOR_PATH = "../predicting_domino_toppling/data/latest/samples-3D-classifier.pkl"
validator = joblib.load(VALIDATOR_PATH)
#  estimator = partial(nprev_estimator, nprev=0)
estimator = combined_estimators
simulator = partial(physics_based, ts=1/1000)
SPLINES_PATH = "../domino_design/data/latest/candidates.pkl"
with open(SPLINES_PATH, 'rb') as f:
    splines = pickle.load(f)
    # S1
    S1 = splines[5]
    # S2
    S2 = splines[18]
    S2[1][0] *= -1
    S2[1][0] += 0.48
    S2[1][1] -= .1


def get_validities(u, spline):
    return validator.predict(
            get_rel_coords(u, spline) / (X_MAX, Y_MAX, A_MAX))


def get_robustness(u, spline):
    return validator.decision_function(
            get_rel_coords(u, spline) / (X_MAX, Y_MAX, A_MAX)).mean()


def compute_size_interval(spline):
    length = spl.arclength(spline)
    nmin = math.floor(length / MAX_SPACING)
    nmax = math.floor(length / MIN_SPACING)

    def valid(n):
        return simulator(equal_spacing(spline, n), spline).max() < np.inf
    try:
        nmin = next(n for n in range(nmin, nmax+1) if valid(n))
    except StopIteration:
        return ()
    try:
        nmax = next(n for n in range(nmax, nmin, -1) if valid(n))
    except StopIteration:
        return nmin,
    return nmin, nmax


def compute_time_interval(spline, nmin, nmax):
    times = np.array([simulator(equal_spacing(spline, n), spline).max()
             for n in range(nmin, nmax)])
    return min(times), max(times[np.isfinite(times)])


def init_optim(spline, time, nmin, nmax):
    # There's a limited number of valid domino run sizes so we just use brute
    # force.
    t_init = np.inf
    for ni in range(nmin, nmax+1):
        ui = equal_spacing(spline, ni)
        ti = simulator(ui, spline).max()
        vi = get_validities(ui, spline)
        if vi.all() and abs(time - ti) < abs(time - t_init):
            t_init = ti
            u_init = ui
    assert t_init != np.inf, "Could not find a valid initial distribution"

    return u_init


def optimize_for_time(spline, time, u_init):
    u_full = u_init.copy()

    def objective(u):
        u_full[1:-1] = u
        return ((time - estimator(u_full, spline).max())**2
                - ALPHA * get_robustness(u_full, spline))

    def umin(u):
        assert (u_full[1:-1] == u).all()
        return np.diff(u_full)

    def umax(u):
        return 1 - u

    cons = (umin, umax)
    rhobeg = 1 / (2 * len(u_init))
    u_opt = opt.fmin_cobyla(objective, u_init[1:-1], cons, rhobeg=rhobeg,
                            disp=0)
    u_full[1:-1] = u_opt

    return u_full


def optimize_for_same_time(s1, s2):
    # Determine time intervals.
    nitv1 = compute_size_interval(s1)
    assert len(nitv1) > 1, "No valid distribution could be found for S1"
    titv1 = compute_time_interval(s1, *nitv1)
    nitv2 = compute_size_interval(s2)
    assert len(nitv2) > 1, "No valid distribution could be found for S2"
    titv2 = compute_time_interval(s2, *nitv2)
    print("Size interval for S1: ", nitv1)
    print("Size interval for S2: ", nitv2)
    print("Time interval for S1: ", titv1)
    print("Time interval for S2: ", titv2)
    # Determine a common time.
    titv12 = interval[titv1[0], titv1[1]] & interval[titv2[0], titv2[1]]
    assert len(titv12), "No common reachable time could be found for S1 and S2"
    t12 = titv12.midpoint[0].inf
    # Initialization: determine the number of dominoes.
    print("Expected time: ", t12)
    u1 = init_optim(s1, t12, *nitv1)
    u2 = init_optim(s2, t12, *nitv2)

    # Optimize robustness
    u1 = optimize_for_time(s1, estimator(u1, s1).max(), u1)
    u2 = optimize_for_time(s2, estimator(u2, s2).max(), u2)
    #  u1 = optimize_for_time(s1, t12, u1)
    #  u2 = optimize_for_time(s2, t12, u2)

    #  eps = .01
    #  while abs(t2 - t1) > eps:
    #      t12 = (t1 + t2) / 2
    #      print("New expected time: ", t12)
    #      u1 = optimize_for_time(s1, t12, u1)
    #      u2 = optimize_for_time(s2, t12, u2)

    #      t1 = estimator(u1, s1).max()
    #      t2 = estimator(u2, s2).max()

    return u1, u2


def export_domino_run(filename, u, spline, sheetsize=(21, 29.7)):
    xy = np.column_stack(spl.splev(u, spline)) * 100
    a = spl.splang(u, spline, degrees=True)
    size = np.tile((t*100, w*100), (len(u), 1))

    extents = xy.ptp(axis=0) + w*100
    assert extents.size == 2
    if extents[1] < sheetsize[0] < extents[0] < sheetsize[1]:
        rot = np.array(((0, 1), (-1, 0)))
        xy = xy.dot(rot)
        a += 90
        extents = extents[::-1]
    xy = xy - (xy.min(axis=0) + xy.max(axis=0))/2 + np.asarray(sheetsize)/2


    cont = export.svg_get_context(filename, (21, 29.7))
    export.svg_add_rectangles(cont, xy, a, size)
    cont.save()


def run_xp2(s1, s2):
    #  u1_opt, u2_opt = optimize_for_same_time(s1, s2)
    #  t1_final = simulator(u1_opt, s1).max()
    #  t2_final = simulator(u2_opt, s2).max()
    #  print("Final times: ", t1_final, t2_final)

    u1_naive = equal_spacing(s1, 25)
    u2_naive = equal_spacing(s2, 25)
    #  show_dominoes([u1_naive, u2_naive], [s1, s2])
    #  show_dominoes([u1_opt, u2_opt], [s1, s2])

    export_domino_run("D1_naive.svg", u1_naive, s1)
    export_domino_run("D2_naive.svg", u2_naive, s2)


def run_xp1(spline):
    # Populate the spline with dominoes.
    u_init = batch_classif_based(spline, batchsize=1)
    t_init = estimator(u_init, spline).max()
    print("Initial number of dominoes: ", len(u_init))
    print("Initial predicted time: ", t_init)

    # Run a simulation to evaluate the toppling time.
    #  times = simulator(u_init, spline)
    #  toppling_time = np.max(times)
    #  print("Initial simulated time: ", toppling_time)
    #  print(np.diff(times))
    #  print(get_rel_coords(u_init, spline) / (X_MAX, Y_MAX, A_MAX))
    #  print(np.diff(estimator(u_init, spline)))
    #  assert toppling_time != np.inf, "Couldn't find valid distrib for input."
    #  import matplotlib.pyplot as plt
    #  plt.plot(np.diff(times))
    #  plt.plot(np.diff(estimator(u_init, spline)))
    #  plt.show()

    # Run optimization to complete the run in a fraction of that time.
    target_time = TIME_FRAC * t_init
    print("Target time: ", target_time)
    # Initialization: determine the number of dominoes.
    u_init = init_optim(spline, target_time)
    u_opt = optimize_for_time(spline, target_time, u_init)
    final_time = estimator(u_opt, spline).max()
    print("Final time: ", final_time)

    # Run a new simulation to validate that new time.
    #  new_toppling_time = np.max(simulator(u_opt, spline))
    #  print("Final simulated time: ", new_toppling_time)
    #  show_dominoes([u_init], [spline])
    show_dominoes([u_opt], [spline])


def main():
    #  if len(sys.argv) < 3:
    #      print(__doc__)
    #      return
    #  spath = sys.argv[1]
    #  sid = int(sys.argv[2])
    #  with open(spath, 'rb') as f:
    #      spline = pickle.load(f)[sid]

    #  u = np.linspace(0, 1, 100)
    #  fig, ax = plt.subplots()
    #  ax.plot(*spl.splev(u, S1), label="S1")
    #  ax.plot(*spl.splev(u, S2), label="S2")
    #  ax.set_aspect('equal', 'datalim')
    #  plt.legend()
    #  plt.show()

    #  run_xp1(S1)
    run_xp2(S1, S2)


if __name__ == "__main__":
    main()
