"""
Visualize energy for a given spline and previous domino position.

Parameters
----------
spath : string
  Path to the list of splines.
sid : int
  Index of the spline.
uprev : float in [0, 1), optional
  Parametric position of the previous domino.

"""
from math import cos, pi, sin, sqrt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box
from sklearn.externals import joblib

from config import t, w, h, X_MAX, Y_MAX, A_MAX, SVC_PATH, SVC2_PATH
from methods import _init_routines_vec
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl  # flake8: noqa E402


def get_inc_classif_based_energy(u, uprev_id, spline):
    svc = joblib.load(SVC_PATH)

    splev, splang, *_, habs, umin, umax, tilted_overlap = _init_routines_vec(
            [u[uprev_id]], spline)

    def objective(ui):
        # Get local Cartesian coordinates
        # Change origin
        x0, y0 = splev(u[uprev_id])
        xi, yi = splev(ui)
        xi = xi - x0
        yi = yi - y0
        # Rotate by -a0
        a0 = splang(u[uprev_id])
        c0 = cos(a0)
        s0 = sin(a0)
        xi = xi*c0 + yi*s0
        yi = -xi*s0 + yi*c0
        # Get relative angle
        ai = (splang(ui) - a0) * 180 / pi
        ai = (ai + 180) % 360 - 180
        # Symmetrize wrt the Ox axis
        ai = np.copysign(ai, yi)
        yi = abs(yi)
        # Normalize
        xi /= X_MAX
        yi /= Y_MAX
        ai /= A_MAX
        # Evaluate
        return svc.decision_function(np.column_stack([xi, yi, ai]))

    f = objective(u)
    c1 = np.concatenate([tilted_overlap([ui]) for ui in u])
    c2 = np.concatenate([umin([ui]) for ui in u])

    return f/20+c1+c2


def get_inc_classif_based_v2_energy(u, uprev_id, spline):
    svc = joblib.load(SVC2_PATH)

    def splev(ui):
        return spl.splev(ui, spline)

    def splang(ui):
        return spl.splang(ui, spline, degrees=False)

    def objective(ui):
        # Get local Cartesian coordinates
        # Change origin
        x0, y0 = splev(u[uprev_id])
        xi, yi = splev(ui)
        xi = xi - x0
        yi = yi - y0
        # Rotate by -a0
        a0 = splang(u[uprev_id])
        c0 = cos(a0)
        s0 = sin(a0)
        xi = xi*c0 + yi*s0
        yi = -xi*s0 + yi*c0
        # Get relative angle
        ai = (splang(ui) - a0) * 180 / pi
        ai = (ai + 180) % 360 - 180
        # Normalize
        xi /= X_MAX
        yi /= Y_MAX
        ai /= A_MAX
        # Evaluate
        return svc.decision_function(np.column_stack([xi, yi, ai]))

    f = objective(u)

    return f


def visualize_energy(uprev, spline):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    npts = 1000
    uprev_id = int(uprev * npts)
    u = np.linspace(0, 1, npts)
    x, y = spl.splev(u, spline)
    energy = get_inc_classif_based_energy(u, uprev_id, spline)
    #  energy = get_inc_classif_based_v2_energy(u, uprev_id, spline)

    plot = ax.scatter(x, y, c=energy, s=2, cmap='viridis')

    base = box(-t * .5, -w * .5, t * .5,  w * .5)
    b1 = translate(rotate(base, spl.splang(u[uprev_id], spline)),
                   x[uprev_id], y[uprev_id])
    ax.plot(*np.array(b1.exterior.coords).T)
    sprev = spl.arclength(spline, uprev)
    margin = int((spl.arclength_inv(spline, sprev + h) - uprev) * npts)
    max_id = energy[uprev_id:uprev_id+margin].argmax() + uprev_id
    b2 = translate(rotate(base, spl.splang(u[max_id], spline)),
                   x[max_id], y[max_id])
    ax.plot(*np.array(b2.exterior.coords).T)
    ax.autoscale_view()
    ax.set_title("The goal is to maximize the energy. E > 0 = good.")
    plt.colorbar(plot)
    plt.ioff()
    plt.show()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])
    uprev = 0. if len(sys.argv) == 3 else float(sys.argv[3])

    with open(spath, 'rb') as f:
        splines = joblib.load(f)
    spline = splines[sid]
    visualize_energy(uprev, spline)


if __name__ == "__main__":
    main()
