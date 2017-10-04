"""
View a domino run.

Parameters
----------
splpath : string
  Path to the list of splines
dompath : string
  Path to the list of domino runs
did : int
  Index of the domino run

"""
import os
import pickle
import sys

from matplotlib import cm
import numpy as np
from panda3d.core import Vec3

sys.path.insert(0, os.path.abspath("../.."))
from primitives import DominoMaker, Floor
from viewers import PhysicsViewer
import spline2d as spl

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(".."))
from domino_design.config import t, w, h, density
from domino_design.evaluate import get_toppling_angle
from predicting_domino_toppling.functions import tilt_box_forward


def show_dominoes(distribs, splines):
    # Viewer
    app = PhysicsViewer()
    app.cam_distance = 5
    app.min_cam_distance = 1
    app.camLens.set_near(0.1)
    app.zoom_speed = 1
    app.set_frame_rate_meter(True)
    # Floor
    floor_path = app.models.attach_new_node("floor")
    floor = Floor(floor_path, app.world)
    floor.create()
    # Domino distributions
    toppling_angle = get_toppling_angle()
    mass = density * t * w * h
    for i, (distrib, spline) in enumerate(zip(distribs, splines)):
        domino_run_np = app.models.attach_new_node("domrun_{}".format(i))
        domino_factory = DominoMaker(domino_run_np, app.world)
        distrib = np.asarray(distrib)
        positions = spl.splev3d(distrib, spline, .5*h)
        headings = spl.splang(distrib, spline)
        for j, (pos, head) in enumerate(zip(positions, headings)):
            domino_factory.add_domino(
                    Vec3(*pos), head, Vec3(t, w, h), mass=mass,
                    prefix="domino_{}".format(j))
        # Initial state
        d_init = domino_run_np.get_child(0)
        tilt_box_forward(d_init, toppling_angle)
        d_init.node().set_transform_dirty()
        # Visual indicators
        color = cm.hsv(np.random.random())
        for domino in domino_run_np.get_children():
            domino.set_color(color)
        v = np.linspace(0., 1., 100)
        spl.show_spline2d(app.render, spline, v, "path_{}".format(i),
                          color=color)

    # Useful if you have to run several in a row.
    try:
        app.run()
    except SystemExit:
        app.destroy()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    splpath = sys.argv[1]
    dompath = sys.argv[2]
    inds = [int(ind) for ind in sys.argv[3:]]

    with open(splpath, 'rb') as fs:
        splines = pickle.load(fs)
    splines = [splines[ind] for ind in inds]
    domruns = np.load(dompath)
    distribs = [domruns['arr_{}'.format(ind)] for ind in inds]

    show_dominoes(distribs, splines)


if __name__ == "__main__":
    main()
