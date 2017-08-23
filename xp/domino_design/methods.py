"""
Various methods used to distribute dominoes along a path.

1. Equal spacing
2. Minimal spacing
3. Incremental classifier based
4. Incremental physically based random search
"""
from functools import lru_cache
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
    return (equal_spacing, minimal_spacing, inc_classif_based,
            inc_physbased_randsearch)


def _init_routines(u, spline):
    """Initialize a bunch of routines likely to be used by several methods."""

    # Define convenience functions.

    @lru_cache()
    def splev(ui):
        return spl.splev(ui, spline)

    @lru_cache()
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
    base = box(-t * .5, -w * .5, t * .5,  w * .5)
    t_t = t / math.sqrt(1 + (t / h)**2)  # t*cos(arctan(theta))
    base_t = box(-t_t * .5, -w * .5, t_t * .5,  w * .5)  # Proj. of tilted base

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
        init_step = t / length
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


def inc_classif_based(spline, init_step=-1, max_ndom=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if init_step == -1:
        init_step = t / length
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


def inc_physbased_randsearch(spline, max_ndom=-1, max_ntrials=-1):
    u = [0.]
    # Default values
    length = spl.arclength(spline)
    if max_ndom == -1:
        max_ndom = int(length / t)
    if max_ntrials == -1:
        max_ntrials = 50  # number of trials per step

    # Start main routine
    last_step = 0
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        ulast = u[-1]
        umin = ulast + 2 * t / length  # at least twice the thickness
        umax = ulast + h * (4 / 9) / length  # hit next domino above 8/9
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


def global_randsearch(spline):
    pass


def global_classif_based(spline):
    pass
