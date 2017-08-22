"""
View a domino run.

Parameters
----------
did : int
  Index of the domino run
splpath : string
  Path to the list of splines
dompath : string
  Path to the list of domino runs

"""
import os
import pickle
import sys

import numpy as np
from panda3d.core import Vec3
from panda3d.core import Mat3

from config import t, w, h
sys.path.insert(0, os.path.abspath("../.."))
from primitives import DominoMaker
from primitives import Floor
from viewers import PhysicsViewer
import spline2d as spl


def show_dominoes(u, spline):
    app = PhysicsViewer()
    app.cam_distance = 5
    app.min_cam_distance = 1
    app.camLens.set_near(0.1)
    app.zoom_speed = 1
    # Floor
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
    spl.show_spline2d(app.render, spline, v, "smoothed path",
                      color=(1, 0, 1, 1))

    app.set_frame_rate_meter(True)
    #  app.finalizeExit = app.destroy  # Replace call to sys.exit with destroy
    app.run()


def main():
    if len(sys.argv) < 4:
        print("Please provide an index and the necessary paths.")
        return
    did = int(sys.argv[1])
    splpath = sys.argv[2]
    dompath = sys.argv[3]

    with open(splpath, 'rb') as fs:
        splines = pickle.load(fs)
    spline = splines[did]
    with open(dompath, 'rb') as fd:
        domruns = pickle.load(fd)
    u = domruns[did]
    print("Number of dominoes: ", len(u))
    show_dominoes(u, spline)


if __name__ == "__main__":
    main()
