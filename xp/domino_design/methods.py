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

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl

from .config import t, w, h
from .config import SVC_PATH, SVC2_PATH
from .config import X_MAX, Y_MAX, A_MAX
from .evaluate import run_simu, setup_dominoes, test_all_toppled
from .evaluate import test_no_successive_overlap_fast


def get_methods():
    """Returns a list of all the available methods."""
    inc_classif_based_v1 = partial(batch_classif_based, batchsize=1)
    batch_classif_based_2 = partial(batch_classif_based, batchsize=2)
    batch_classif_based_3 = partial(batch_classif_based, batchsize=3)
    return (equal_spacing,
            minimal_spacing,
            inc_physbased_randsearch,
            inc_classif_based_v1,
            inc_classif_based_v2,
            batch_classif_based_2,
            batch_classif_based_3,
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

    def tilted_overlap(ui):
        u1, u2 = u[-1], float(ui)
        # Define first rectangle (projection of tilted base)
        a1 = splang(u1)
        h1_rad = a1 * math.pi / 180
        c1_t = (np.hstack(splev(u1))
                + .5 * (t + t_t)
                * np.array([math.cos(h1_rad), math.sin(h1_rad)]))
        b1_t = translate(rotate(base_t, a1), c1_t[0], c1_t[1])
        # Define second rectangle
        a2 = splang(u2)
        c2 = np.hstack(splev(u2))
        b2 = translate(rotate(base, a2), c2[0], c2[1])

        # --For debug--
        #  import matplotlib.pyplot as plt
        #  fig, ax = plt.subplots()
        #  ax.set_aspect('equal')
        #  ax.plot(*np.array(b1_t.exterior.coords).T, label='D1')
        #  ax.plot(*np.array(b2.exterior.coords).T, label='D2')
        #  ax.plot(*spl.splev(np.linspace(0, 1), spline))
        #  plt.legend()
        #  plt.ioff()
        #  plt.show()

        # Return intersection
        return - b1_t.intersection(b2).area / (t_t * w)

    return splev, splang, xmin, xmax, yabs, habs, umin, umax, tilted_overlap


def _init_routines_vec(u, spline):

    # Convenience functions

    def splev(ui):
        return spl.splev(ui, spline)

    def splang(ui):
        return spl.splang(ui, spline, degrees=False)

    # Constraints

    def xmin(ui):
        ui = np.concatenate(([u[-len(ui)]], ui))
        xi, yi = splev(ui)
        xi = xi[1:] - xi[:-1]
        yi = yi[1:] - yi[:-1]
        ai = splang(ui[:-1])
        xi = xi*np.cos(ai) + yi*np.sin(ai)
        return xi - t

    def xmax(ui):
        ui = np.concatenate(([u[-len(ui)]], ui))
        xi, yi = splev(ui)
        xi = xi[1:] - xi[:-1]
        yi = yi[1:] - yi[:-1]
        ai = splang(ui[:-1])
        xi = xi*np.cos(ai) + yi*np.sin(ai)
        return h - xi

    def yabs(ui):
        ui = np.concatenate(([u[-len(ui)]], ui))
        xi, yi = splev(ui)
        xi = xi[1:] - xi[:-1]
        yi = yi[1:] - yi[:-1]
        ai = splang(ui[:-1])
        yi = -xi*np.sin(ai) + yi*np.cos(ai)
        return w - abs(yi)

    def habs(ui):
        ai = splang(np.concatenate(([u[-len(ui)]], ui)))
        diff = (ai[1:] - ai[:-1] + 180) % 360 - 180
        return 45 - abs(diff)

    def umin(ui):
        diff = ui - np.concatenate(([u[-len(ui)]], ui[:-1]))
        diff[diff > 0] = 0
        return diff

    def umax(ui):
        diff = 1 - np.asarray(ui)
        diff[diff > 0] = 0
        return diff

    # Define variables for non-overlap constraint
    base = box(-t/2, -w/2, t/2,  w/2)
    t_tilt = t / math.sqrt(1 + (t / h)**2)  # t*cos(arctan(theta))
    base_tilt = box(-t_tilt/2, -w/2, t_tilt/2,  w/2)  # Project. of tilted base

    def no_overlap(ui):
        ui = np.concatenate(([u[-len(ui)]], ui))
        ai = splang(ui)
        ci = np.column_stack(splev(ui))
        ci_tilt = ci + .5 * (t + t_tilt) * np.column_stack(
                (np.cos(ai), np.sin(ai)))
        # Create projections of tilted bases (no need to tilt the last one)
        bi_tilt = [
                translate(rotate(base_tilt, aij, use_radians=True), *cij_tilt)
                for aij, cij_tilt in zip(ai[:-1], ci_tilt[:-1])]
        # Create projections of untilted bases
        bi = [translate(rotate(base, aik, use_radians=True), *cik)
              for aik, cik in zip(ai, ci)]

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

        # Return intersections
        isect = np.array([- bij.intersection(bik).area / (t * w)
                          for bij, bik in zip(bi[:-1], bi[1:])])
        isect_tilt = np.array([- bij_tilt.intersection(bij).area / (t_tilt * w)
                               for bij_tilt, bij in zip(bi_tilt, bi[1:])])
        return isect + isect_tilt

    return splev, splang, xmin, xmax, yabs, habs, umin, umax, no_overlap


def equal_spacing(spline, ndom=-1):
    u = [0.]
    # Default value
    length = spl.arclength(spline)
    if ndom == -1:
        ndom = math.floor(length / (h / 3))

    s = np.linspace(0, length, ndom)[1:]
    u.extend(spl.arclength_inv(spline, s))

    return u


def minimal_spacing(spline, init_step=-1, max_ndom=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if init_step == -1:
        init_step = np.asscalar(spl.arclength_inv(spline, h/3))
    if max_ndom == -1:
        max_ndom = int(length / t)
    # Constraints
    splev, splang, *_, umin, umax, no_overlap = _init_routines_vec(u, spline)
    cons = (no_overlap, umin, umax)
    # Objective

    def objective(ui):
        return (ui - u[-1])**2

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        init_step = last_step if last_step else init_step
        init_guess = np.atleast_1d(u[-1] + init_step)
        unew = opt.fmin_cobyla(objective, init_guess, cons,
                               rhobeg=init_step/100, disp=0)
        # Early termination condition
        if not test_no_successive_overlap_fast((u[-1], unew), spline):
            print("New sample too close to the previous; terminating.")
            break
        u.append(np.asscalar(unew))
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
            if test_no_successive_overlap_fast((ulast, unew), spline):
                doms_np = run_simu(*setup_dominoes(u + [unew], spline))
                if test_all_toppled(doms_np):
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
        a0 = splang(u[-1])
        xi, yi = splev(float(ui))
        ai = splang(float(ui))
        xi = float(xi - x0)
        yi = float(yi - y0)
        ai = float(ai - a0)
        # Symmetrize
        ai = math.copysign(ai, yi)
        yi = abs(yi)
        # Normalize
        xi /= X_MAX
        yi /= Y_MAX
        ai /= A_MAX
        # Evaluate
        return -svc.decision_function([[xi, yi, ai]])[0]

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        init_guess = last_step if last_step else init_step
        unew = opt.fmin_cobyla(objective, u[-1]+init_guess, cons,
                               rhobeg=init_step, disp=0)
        # Early termination condition
        if not test_no_successive_overlap_fast((u[-1], unew), spline):
            print("New sample too close to the previous; terminating.")
            break
        u.append(float(unew))
        last_step = u[-1] - u[-2]

    return u


def inc_classif_based_v2(spline, init_step=-1, max_ndom=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if init_step == -1:
        init_step = np.asscalar(spl.arclength_inv(spline, h/3))
    if max_ndom == -1:
        max_ndom = int(length / t)
    # Constraints
    splev, splang, *_, umin, umax, _ = _init_routines_vec(u, spline)
    cons = (umin, umax)
    # Objective
    svc = joblib.load(SVC2_PATH)
    energy = svc.decision_function

    def objective(ui):
        # Get local Cartesien coordinates
        # Change origin
        x0, y0 = splev(u[-1])
        xi, yi = splev(ui)
        xi = xi - x0
        yi = yi - y0
        # Rotate by -a0
        a0 = splang(u[-1])
        c0 = np.cos(a0)
        s0 = np.sin(a0)
        xi = xi*c0 + yi*s0
        yi = -xi*s0 + yi*c0
        # Get relative angle
        ai = np.degrees(splang(ui) - a0)
        ai = (ai + 180) % 360 - 180  # Convert from [0, 360) to [-180, 180)
        # Normalize
        xi /= X_MAX
        yi /= Y_MAX
        ai /= A_MAX
        # Evaluate
        return -energy(np.column_stack([xi, yi, ai]))

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        init_step = last_step if last_step else init_step
        init_guess = np.atleast_1d(u[-1] + init_step)
        unew = opt.fmin_cobyla(objective, init_guess, cons,
                               rhobeg=init_step/100, disp=0)
        # Early termination condition
        if not test_no_successive_overlap_fast((u[-1], unew), spline):
            print("New sample too close to the previous; terminating.")
            break
        u.append(np.asscalar(unew))
        last_step = u[-1] - u[-2]

    return u


def batch_classif_based(spline, batchsize=2, init_step=-1, max_ndom=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if init_step == -1:
        init_step = np.asscalar(spl.arclength_inv(spline, h/3))
    if max_ndom == -1:
        max_ndom = int(length / t)
    # Constraints
    splev, splang, *_, umin, umax, no_overlap = _init_routines_vec(u, spline)
    cons = (no_overlap, umin, umax)
    # Objective
    svc = joblib.load(SVC_PATH)
    energy = svc.decision_function

    def objective(ui):
        if not np.isfinite(ui).all():
            return np.inf
        ui = np.concatenate(([u[-len(ui)]], ui))
        # Get local Cartesian coordinates
        # Change origin
        xi, yi = splev(ui)
        xi = xi[1:] - xi[:-1]
        yi = yi[1:] - yi[:-1]
        # Rotate by -a_i-1
        ai = splang(ui)
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
        # Evaluate
        return -np.mean(energy(np.column_stack((xi, yi, ai))))

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        # Define initial guess
        init_step = last_step if last_step else init_step
        init_guess = [u[-1] + init_step]
        if len(u) >= batchsize:
            # Insert the 'batchsize'-1 previous dominoes at the beginning.
            init_guess[0:0] = u[len(u)-batchsize+1:]
        # Run optimization
        unew = opt.fmin_cobyla(objective, init_guess, cons,
                               rhobeg=init_step/100, disp=0)
        # Early termination condition
        if not test_no_successive_overlap_fast(
                [u[-len(unew)]] + unew.tolist(), spline):
            print("Samples are too close; terminating.")
            break
        # Save result
        if len(u) >= batchsize:
            # Replace the 'batchsize'-1 last values and add the new value.
            u[len(u)-batchsize+1:len(u)] = unew
        else:
            u.append(np.asscalar(unew))

        last_step = u[-1] - u[-2]

    return u


def global_randsearch(spline):
    pass


def global_classif_based(spline):
    pass
