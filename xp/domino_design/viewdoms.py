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

import numpy as np
from panda3d.core import Vec3, Mat3

sys.path.insert(0, os.path.abspath("../.."))
from primitives import DominoMaker, Floor
from viewers import PhysicsViewer
import spline2d as spl

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(".."))
from domino_design.config import t, w, h, density


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
    mass = density * t * w * h
    for i, (pos, head) in enumerate(zip(positions, headings)):
        domino_factory.add_domino(
                Vec3(*pos), head, Vec3(t, w, h), mass=mass,
                prefix="domino_{}".format(i))

    first_domino = domino_run_np.get_child(0)
    first_domino.node().set_angular_velocity(angvel_init)
    v = np.linspace(0., 1., 100)
    spl.show_spline2d(app.render, spline, v, "smoothed path",
                      color=(1, 0, 1, 1))

    app.set_frame_rate_meter(True)
    app.run()
    # Or if you have to open several in the _same_ python run:
    #  try:
    #      app.run()
    #  except SystemExit:
    #      app.destroy()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    splpath = sys.argv[1]
    dompath = sys.argv[2]
    did = int(sys.argv[3])

    with open(splpath, 'rb') as fs:
        splines = pickle.load(fs)
    spline = splines[did]
    domruns = np.load(dompath)
    u = domruns['arr_{}'.format(did)]
    print("Number of dominoes: ", len(u))
    show_dominoes(u, spline)


if __name__ == "__main__":
    main()
