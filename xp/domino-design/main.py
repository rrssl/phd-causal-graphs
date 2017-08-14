"""
Designing domino runs.

"""
from functools import lru_cache
import math
import os
import sys

import numpy as np
import scipy.optimize as opt
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box

from sklearn.externals import joblib

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../domino-learning"))
from config import t, w, h
import spline2d as spl


# Used to resize the path; ratio of the area of the path's bounding box over
# the area of the domino's smallest face.
PATH_SIZE_RATIO = 3
SVC_PATH = "../domino-learning/samples-3D-model.pkl"


def main():
    if len(sys.argv) <= 1:
        print("Please provide a file name for the domino run path.")
        return
    # Load path
    fname = sys.argv[1]
    path = np.load(fname)[0]
    # Translate, resize and smooth the path
    path -= path.min(axis=0)
    path *= math.sqrt(
            PATH_SIZE_RATIO * t * w / (path[:, 0].max() * path[:, 1].max()))
    spline = spl.get_smooth_path(path, s=.1)
    @lru_cache()
    def splev(ui):
        return spl.splev(ui, spline)
    @lru_cache()
    def splang(ui):
        return spl.get_spline_phi(ui, spline)
    # Initialize parameter list, first param value, and initial spacing
    u = [0.]
    length = spl.arclength(spline)
    init_step = t / length
    last_step = 0.
    max_ndom = 2 if True else int(length / t)
    # Define bound constraints
    def xmin(ui):
        return (splev(ui)[0] - splev(u[-1])[0]) - t
    def xmax(ui):
        return 1.5*h - (splev(ui)[0] - splev(u[-1])[0])
    def yabs(ui):
        return w - abs(splev(ui)[1] - splev(u[-1])[1])
    def habs(ui):
        return 90 - abs(splang(ui) - splang(u[-1]))
    # Define non-overlap constraint
    base = box(-t * .5, -w * .5, t * .5,  w * .5)
    t_t = t / math.sqrt(1 + (t / h)**2)  # t*cos(arctan(theta))
    base_t = box(-t_t * .5, -w * .5, t_t * .5,  w * .5)  # Proj. of tilted base
    def tilted_overlap(ui, _debug=False):
        u1, u2 = u[-1], ui
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
        return b1_t.intersection(b2).area / (t_t * w)
    # Define objective
    svc = joblib.load(SVC_PATH)
    def objective(ui):
        u1, u2 = u[-1], ui
        # Get local coordinates
        x1, y1 = splev(u1)
        h1 = splang(u1)
        x2, y2 = splev(u2)
        h2 = splang(u2)
        x = float(x2 - x1)
        y = float(y2 - y1)
        h = h2 - h1
        # Symmetrize
        h = math.copysign(h, y)
        y = abs(y)
        # Evaluate
        return svc.decision_function([[x / (1.5*h), y / w, h / 90]])[0]
    # Start main routine
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        print(tilted_overlap(0.6, True))
        print(objective(0.6))
        break
    # Display resulting domino run


if __name__ == "__main__":
    main()
