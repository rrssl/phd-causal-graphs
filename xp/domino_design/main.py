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
PATH_SIZE_RATIO = 25
SMOOTHING_FACTOR = .1
# "EG" settings
#  PATH_SIZE_RATIO = 80
#  SMOOTHING_FACTOR = .02
SVC_PATH = "../domino-learning/samples-3D-model.pkl"


def visualize_objective(spline, svc):

    def objective(ui, uprev=0.):
        # Get local coordinates
        x0, y0 = spl.splev(uprev, spline)
        h0 = spl.splang(uprev, spline)
        xi, yi = spl.splev(ui, spline)
        hi = spl.splang(ui, spline)
        xi = xi - x0
        yi = yi - y0
        hi = hi - h0
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
    uprev_id = int(0.0 * npts)
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
    ax.scatter(x, y, marker='+')
    ax.autoscale_view()
    plt.colorbar(lc2)
    plt.ioff()
    plt.show()


def show_domino_run(spline, u):
    from panda3d.core import Vec3
    from panda3d.core import Mat3
    from primitives import DominoMaker
    from primitives import Floor
    from viewers import PhysicsViewer

    app = PhysicsViewer()
    app.cam_distance = 10
    app.min_cam_distance = 1.5
    app.zoom_speed = 2
    #Floor
    floor_path = app.models.attach_new_node("floor")
    floor = Floor(floor_path, app.world)
    floor.create()
    # Set initial angular velocity
    # (but maybe we should just topple instead of giving velocity)
    angvel_init = Vec3(0., 15., 0.)
    angvel_init = Mat3.rotate_mat(spl.splang(0, spline)).xform(
            angvel_init)
    # Instantiate dominoes.
    domino_run_np = app.models.attach_new_node("domino_run")
    domino_factory = DominoMaker(domino_run_np, app.world)
    u = np.array(u)
    positions = spl.splev3d(u, spline, .5*h)
    headings = spl.splang(u, spline)
    for i, (pos, head) in enumerate(zip(positions, headings)):
        domino_factory.add_domino(
                Vec3(*pos), head, Vec3(t, w, h), mass=1,
                prefix="domino_{}".format(i))

    first_domino = domino_run_np.get_child(0)
    first_domino.node().set_angular_velocity(angvel_init)
    v = np.linspace(0., 1., 100)
    path = spl.show_spline2d(app.render, spline, v, "smoothed path",
                             color=(1, 0, 1, 1))

    app.set_frame_rate_meter(True)
    app.run()


def main():
    if len(sys.argv) <= 1:
        print("Please provide a file name for the domino run path.")
        return
    # Load path
    fname = sys.argv[1]
    path = np.load(fname)[0]
    # Translate, resize and smooth the path
    path -= path.min(axis=0)
    path *= PATH_SIZE_RATIO * math.sqrt(
            t * w / (path[:, 0].max() * path[:, 1].max()))
    spline = spl.get_smooth_path(path, s=SMOOTHING_FACTOR)
    @lru_cache()
    def splev(ui):
        return spl.splev(ui, spline)
    @lru_cache()
    def splang(ui):
        return spl.splang(ui, spline)
    # Initialize parameter list, first param value, and initial spacing
    u = [0.]
    length = spl.arclength(spline)
    # TODO. Up to there, the method is completely generic. Put the rest in
    # a routine specific to this method!
    init_step = t / length
    last_step = 0
    max_ndom = int(length / t)
    # Define bound constraints
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
    # Define non-overlap constraint
    base = box(-t * .5, -w * .5, t * .5,  w * .5)
    t_t = t / math.sqrt(1 + (t / h)**2)  # t*cos(arctan(theta))
    base_t = box(-t_t * .5, -w * .5, t_t * .5,  w * .5)  # Proj. of tilted base
    def tilted_overlap(ui, _debug=False):
        # TODO. Put me out of main() so I can be reused elsewhere!
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
        return -b1_t.intersection(b2).area / (t_t * w)
    # Define objective
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

    #  visualize_objective(spline, svc)
    #  return

    # Start main routine
    #  cons = (xmin, xmax, yabs, habs, umax, tilted_overlap)
    #  cons = (xmin, xmax, umin, umax, tilted_overlap)
    cons = (tilted_overlap, umin, umax, habs)
    while 1. - u[-1] > last_step and len(u) < max_ndom:
        init_guess = last_step if last_step else init_step
        unew = opt.fmin_cobyla(objective, u[-1]+init_guess, cons,
                               rhobeg=init_step)
        if abs(unew - u[-1]) < init_step / 10:
            print("New sample too close to the previous; terminating.")
            break
        u.append(float(unew))
        last_step = u[-1] - u[-2]
    print(u)
    # Display resulting domino run
    show_domino_run(spline, u)


if __name__ == "__main__":
    main()
