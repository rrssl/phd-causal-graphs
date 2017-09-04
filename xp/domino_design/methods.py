"""
Various methods used to distribute dominoes along a path.

1. Equal spacing
2. Minimal spacing
3. Incremental physically based random search
4. Incremental classifier based
5. Batch classifier based
"""
from functools import partial
import math
import os
import random
import sys

import numpy as np
import scipy.optimize as opt
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box
from sklearn.externals import joblib

from config import t, w, h
from config import SVC_PATH
from evaluate import test_all_topple
from evaluate import test_no_overlap
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


def get_methods():
    """Returns a list of all the available methods."""
    batch_classif_based_2 = batch_classif_based
    batch_classif_based_3 = partial(batch_classif_based, batchsize=3)
    batch_classif_based_4 = partial(batch_classif_based, batchsize=4)
    batch_classif_based_5 = partial(batch_classif_based, batchsize=5)
    return (equal_spacing,
            minimal_spacing,
            inc_physbased_randsearch,
            inc_classif_based,
            batch_classif_based_2,
            batch_classif_based_3,
            batch_classif_based_4,
            batch_classif_based_5,
            )


def printf(f):
    def wrap(*args, **kwargs):
        out = f(*args, **kwargs)
        print(f.__name__, out)
        return out
    return wrap


def _init_routines(u, spline):
    """Initialize a bunch of routines likely to be used by several methods."""

    # Define convenience functions.

    def splev(ui):
        return spl.splev(ui, spline)

    def splang(ui):
        return spl.splang(ui, spline)

    # Define possible constraints (not all of them are used in every method).

    def xmin(ui):
        return abs(splev(float(ui))[0] - splev(u[-1])[0]) - t

    def xmax(ui):
        return 1.0*h - abs(splev(float(ui))[0] - splev(u[-1])[0])

    def yabs(ui):
        return w - abs(splev(float(ui))[1] - splev(u[-1])[1])

    def habs(ui):
        diff = splang(float(ui)) - splang(u[-1])
        diff = (diff + 180) % 360 - 180
        return 45 - abs(diff)

    def umin(ui):
        return ui - u[-1]

    def umax(ui):
        return 1 - ui

    # Define variables for non-overlap constraint
    base = box(-t/2, -w/2, t/2,  w/2)
    t_t = t / math.sqrt(1 + (t / h)**2)  # t*cos(arctan(theta))
    base_t = box(-t_t/2, -w/2, t_t/2,  w/2)  # Proj. of tilted base

    def tilted_overlap(ui, _debug=False):
        u1, u2 = u[-1], float(ui)
        # Define first rectangle (projection of tilted base)
        h1 = splang(u1)
        h1_rad = h1 * math.pi / 180
        c1_t = (np.hstack(splev(u1))
                + .5 * (t + t_t)
                * np.array([math.cos(h1_rad), math.sin(h1_rad)]))
        b1_t = translate(rotate(base_t, h1), c1_t[0], c1_t[1])
        # Define second rectangle
        h2 = splang(u2)
        c2 = np.hstack(splev(u2))
        b2 = translate(rotate(base, h2), c2[0], c2[1])

        if _debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.plot(*np.array(b1_t.exterior.coords).T, label='D1')
            ax.plot(*np.array(b2.exterior.coords).T, label='D2')
            ax.plot(*spl.splev(np.linspace(0, 1), spline))
            plt.legend()
            plt.ioff()
            plt.show()

        # Return intersection
        return - b1_t.intersection(b2).area / (t_t * w)

    return splev, splang, xmin, xmax, yabs, habs, umin, umax, tilted_overlap


def _init_routines_vec(u, spline):

    # Convenience functions

    def splev(ui):
        return spl.splev(ui, spline)

    def splang(ui):
        return spl.splang(ui, spline)

    # Constraints

    def xmin(ui):
        xi = splev(np.concatenate(([u[-len(ui)]], ui)))[0]
        return abs(xi[1:] - xi[:-1]) - t

    def xmax(ui):
        xi = splev(np.concatenate(([u[-len(ui)]], ui)))[0]
        return h - abs(xi[1:] - xi[:-1])

    def yabs(ui):
        yi = splev(np.concatenate(([u[-len(ui)]], ui)))[1]
        return w - abs(yi[1:] - yi[:-1])

    def habs(ui):
        hi = splang(np.concatenate(([u[-len(ui)]], ui)))
        diff = (hi[1:] - hi[:-1] + 180) % 360 - 180
        return 45 - abs(diff)

    def umin(ui):
        return ui - np.concatenate(([u[-len(ui)]], ui[:-1]))

    def umax(ui):
        return 1 - np.asarray(ui)

    # Define variables for non-overlap constraint
    base = box(-t/2, -w/2, t/2,  w/2)
    t_tilt = t / math.sqrt(1 + (t / h)**2)  # t*cos(arctan(theta))
    base_tilt = box(-t_tilt/2, -w/2, t_tilt/2,  w/2)  # Project. of tilted base

    def tilted_overlap(ui):
        ui = np.concatenate(([u[-len(ui)]], ui))
        hi = spl.splang(ui, spline, degrees=False)
        ci = np.column_stack(splev(ui))
        ci_tilt = ci + .5 * (t + t_tilt) * np.column_stack(
                (np.cos(hi), np.sin(hi)))
        # Define previous rectangles (projection of tilted base)
        bi_tilt = [
                translate(rotate(base_tilt, hij, use_radians=True), *cij_tilt)
                for hij, cij_tilt in zip(hi[:-1], ci_tilt[:-1])]
        # Define next rectangles
        bi = [translate(rotate(base, hik, use_radians=True), *cik)
              for hik, cik in zip(hi[1:], ci[1:])]

        # --For debug--
        #  import matplotlib.pyplot as plt
        #  fig, ax = plt.subplots()
        #  ax.set_aspect('equal')
        #  ax.plot(*np.array(bi_tilt[0].exterior.coords).T, label='D1')
        #  ax.plot(*np.array(bi[0].exterior.coords).T, label='D2')
        #  ax.plot(*spl.splev(np.linspace(0, 1), spline))
        #  plt.legend()
        #  plt.ioff()
        #  plt.show()

        # Return intersection
        return np.array([- bij_tilt.intersection(bij).area / (t_tilt * w)
                         for bij_tilt, bij in zip(bi_tilt, bi)])

    return splev, splang, xmin, xmax, yabs, habs, umin, umax, tilted_overlap


def equal_spacing(spline, ndom=-1):
    u = [0.]
    # Default value
    length = spl.arclength(spline)
    if ndom == -1:
        ndom = int(length / (t * 2))

    s = np.linspace(0, length, ndom)[1:]
    u.extend(spl.arclength_inv(spline, s))

    return u


def minimal_spacing(spline, init_step=-1, max_ndom=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if init_step == -1:
        init_step = spl.arclength_inv(spline, t)
    if max_ndom == -1:
        max_ndom = int(length / t)
    # Constraints
    splev, splang, *_, umin, umax, tilted_overlap = _init_routines(u, spline)
    cons = (tilted_overlap, umin, umax)
    # Objective

    def objective(ui):
        return (ui - u[-1])**2

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        init_guess = last_step if last_step else init_step
        unew = opt.fmin_cobyla(objective, u[-1]+init_guess, cons,
                               rhobeg=init_step, disp=0)
        if abs(unew - u[-1]) < init_step / 10:
            print("New sample too close to the previous; terminating.")
            break
        u.append(float(unew))
        last_step = u[-1] - u[-2]

    return u


def inc_physbased_randsearch(spline, max_ndom=-1, max_ntrials=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if max_ndom == -1:
        max_ndom = int(length / t)
    if max_ntrials == -1:
        max_ntrials = 100  # number of trials per step

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        ulast = u[-1]
        slast = spl.arclength(spline, ulast)
        umin = spl.arclength_inv(spline, slast + t)
        umax = spl.arclength_inv(
                spline, slast + h * (4 / 9))  # hit next domino above 8/9
        umax = min(umax, 1)

        ntrials = 0
        while ntrials < max_ntrials:
            unew = random.uniform(umin, umax)
            if (test_no_overlap((ulast, unew), spline) and
                    test_all_topple(u + [unew], spline)):
                u.append(unew)
                break
            ntrials += 1
        else:
            break  # Last step has failed
        last_step = u[-1] - ulast

    return u


def inc_classif_based(spline, init_step=-1, max_ndom=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if init_step == -1:
        init_step = np.asscalar(spl.arclength_inv(spline, t))
    if max_ndom == -1:
        max_ndom = int(length / t)
    # Constraints
    splev, splang, *_, habs, umin, umax, tilted_overlap = _init_routines(
            u, spline)
    #  cons = (xmin, xmax, yabs, habs, umax, tilted_overlap)
    #  cons = (xmin, xmax, umin, umax, tilted_overlap)
    cons = (tilted_overlap, umin, umax, habs)
    # Objective
    svc = joblib.load(SVC_PATH)

    def objective(ui):
        # Get local coordinates
        x0, y0 = splev(u[-1])
        h0 = splang(u[-1])
        xi, yi = splev(float(ui))
        hi = splang(float(ui))
        xi = float(xi - x0)
        yi = float(yi - y0)
        hi = float(hi - h0)
        # Symmetrize
        hi = math.copysign(hi, yi)
        yi = abs(yi)
        # Normalize
        xi /= 1.5*h
        yi /= w
        hi /= 90
        # Evaluate
        return -svc.decision_function([[xi, yi, hi]])[0]

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        init_guess = last_step if last_step else init_step
        unew = opt.fmin_cobyla(objective, u[-1]+init_guess, cons,
                               rhobeg=init_step, disp=0)
        if abs(unew - u[-1]) < init_step / 10:
            print("New sample too close to the previous; terminating.")
            break
        u.append(float(unew))
        last_step = u[-1] - u[-2]

    return u


def batch_classif_based(spline, batchsize=2, init_step=-1, max_ndom=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if init_step == -1:
        init_step = np.asscalar(spl.arclength_inv(spline, t))
    if max_ndom == -1:
        max_ndom = int(length / t)
    # Constraints
    splev, splang, *_, habs, umin, umax, tilted_overlap = _init_routines_vec(
            u, spline)
    #  cons = (xmin, xmax, yabs, habs, umax, tilted_overlap)
    #  cons = (xmin, xmax, umin, umax, tilted_overlap)
    cons = (tilted_overlap, umin, umax, habs)
    # Objective
    svc = joblib.load(SVC_PATH)

    def objective(ui):
        ui = np.concatenate(([u[-len(ui)]], ui))
        # Get local coordinates
        xi, yi = splev(ui)
        hi = splang(ui)
        xi = xi[1:] - xi[:-1]
        yi = yi[1:] - yi[:-1]
        hi = hi[1:] - hi[:-1]
        # Symmetrize
        hi = np.copysign(hi, yi)
        yi = abs(yi)
        # Normalize
        xi /= 1.5*h
        yi /= w
        hi /= 90
        # Evaluate
        return -min(svc.decision_function(np.column_stack((xi, yi, hi))))

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        # Define initial guess
        init_guess = [u[-1] + last_step if last_step else init_step]
        if len(u) >= batchsize:
            # Insert the 'batchsize'-1 previous dominoes at the beginning.
            init_guess[0:0] = u[len(u)-batchsize+1:]
        # Run optimization
        unew = opt.fmin_cobyla(objective, init_guess, cons, rhobeg=init_step,
                               disp=0)
        # Save result
        if len(u) >= batchsize:
            # Replace the 'batchsize'-1 last values and add the new value.
            u[len(u)-batchsize+1:len(u)] = unew
        else:
            u.append(np.asscalar(unew))
        # Early termination condition
        terminate = False
        if len(u) > batchsize:
            # Dominoes are placed by batch.
            if any(np.isclose(
                    u[-batchsize:], u[-batchsize-1:-1],
                    rtol=0, atol=init_step/10)):
                terminate = True
        else:
            # Dominoes are placed one by one
            if abs(u[-1] - u[-2]) < init_step / 10:
                terminate = True
        if terminate:
            print("Samples are too close; terminating.")
            break

        last_step = u[-1] - u[-2]

    return u


def global_randsearch(spline):
    pass


def global_classif_based(spline):
    pass
