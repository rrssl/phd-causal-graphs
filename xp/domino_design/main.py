"""
Run a method on an input sketch and view its output.

Parameters
----------
mid : int
  Method id. See below.
path : string
  Path to the sketch file.

Methods:
1. Equal spacing
2. Minimal spacing
3. Incremental classifier based
4. Incremental physically based random search

"""
import math
import os
import sys

import numpy as np
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box
from sklearn.externals import joblib

from config import t, w, h, SVC_PATH
from methods import get_methods
from viewdoms import show_dominoes
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


# Used to resize the path; ratio of the area of the path's bounding box over
# the area of the domino's smallest face.
#  PATH_SIZE_RATIO = 25
#  SMOOTHING_FACTOR = .1
# "EG" settings
PATH_SIZE_RATIO = 80
SMOOTHING_FACTOR = .02
# More challenging, use with 20170812-182213.npy
#  PATH_SIZE_RATIO = 15
#  SMOOTHING_FACTOR = .01


def visualize_svc_energy(spline, uprev):
    svc = joblib.load(SVC_PATH)

    def objective(ui, uprev=0.):
        # Get local coordinates
        x0, y0 = spl.splev(uprev, spline)
        h0 = spl.splang(uprev, spline)
        xi, yi = spl.splev(ui, spline)
        hi = spl.splang(ui, spline)
        xi = xi - x0
        yi = yi - y0
        hi = ui - h0
        # Symmetrize wrt the Ox axis
        hi = np.copysign(hi, yi)
        yi = abs(yi)
        # Normalize
        xi /= 1.5*h
        yi /= w
        hi /= 90
        # Evaluate
        return -svc.decision_function(np.column_stack([xi, yi, hi]))

    base = box(-t * .5, -w * .5, t * .5,  w * .5)
    t_t = t / math.sqrt(1 + (t / h)**2)  # t*cos(arctan(theta))
    base_t = box(-t_t * .5, -w * .5, t_t * .5,  w * .5)  # Proj. of tilted base

    def tilted_overlap(ui, uprev=0.):
        u1, u2 = uprev, float(ui)
        # Define first rectangle (projection of tilted base)
        h1 = spl.splang(u1, spline)
        h1_rad = h1 * math.pi / 180
        c1_t = (np.hstack(spl.splev(u1, spline))
                + .5 * (t + t_t)
                * np.array([math.cos(h1_rad), math.sin(h1_rad)]))
        b1_t = translate(rotate(base_t, h1), c1_t[0], c1_t[1])
        # Define second rectangle
        h2 = spl.splang(u2, spline)
        c2 = np.hstack(spl.splev(u2, spline))
        b2 = translate(rotate(base, h2), c2[0], c2[1])
        return b1_t.intersection(b2).area / (t_t * w)

    import matplotlib.pyplot as plt
    from colorline import colorline
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')
    npts = 100
    uprev_id = int(uprev * npts)
    u = np.linspace(0, 1, npts)
    x, y = spl.splev(u, spline)
    f = objective(u[uprev_id+1:], u[uprev_id])
    c = [tilted_overlap(ui, u[uprev_id]) for ui in u[uprev_id+1:]]

    lc1 = colorline(ax, x[:uprev_id+2], y[:uprev_id+2], 0, cmap='autumn',
                    linewidth=3)
    lc2 = colorline(ax, x[uprev_id+1:], y[uprev_id+1:], f+c, cmap='viridis_r',
                    linewidth=3)

    b1 = translate(rotate(base, spl.splang(u[uprev_id], spline)),
                   x[uprev_id], y[uprev_id])
    ax.plot(*np.array(b1.exterior.coords).T)
    #  ax.scatter(x, y, marker='+')
    ax.autoscale_view()
    plt.colorbar(lc2)
    plt.ioff()
    plt.show()


def main():
    if len(sys.argv) < 3:
        print("Please provide a method number and a file name for the sketch.")
        return
    mid = int(sys.argv[1])
    fname = sys.argv[2]
    # Select method
    method = get_methods()[mid-1]
    # Load path
    path = np.load(fname)[0]
    # Translate, resize and smooth the path
    path -= path.min(axis=0)
    path *= PATH_SIZE_RATIO * math.sqrt(
            t * w / (path[:, 0].max() * path[:, 1].max()))
    spline = spl.get_smooth_path(path, s=SMOOTHING_FACTOR)

    u = method(spline)
    show_dominoes(u, spline)


if __name__ == "__main__":
    main()
