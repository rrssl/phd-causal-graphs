"""
View a domino run.

Parameters
----------
splpath : string
  Path to the list of splines
dompath : string
  Path to the list of domino runs
did : sequence of int
  Indexes of the domino runs

"""
from itertools import cycle
import os
import pickle
import sys

from matplotlib import cm
import numpy as np
from panda3d.core import Vec4

sys.path.insert(0, os.path.abspath(".."))
from gui.viewers import PhysicsViewer  # noqa: E402
import core.spline2d as spl  # noqa: E402
from xp.simulate import Simulation  # noqa: E402


class DominoViewer(PhysicsViewer):
    def __init__(self):
        super().__init__()
        self.cam_distance = 5
        self.min_cam_distance = 1
        self.camLens.set_near(0.1)
        self.zoom_speed = 1
        self.set_frame_rate_meter(True)

        self.colors = cycle(cm.hsv(np.linspace(0, 1, 6)))

        self.simu = Simulation(visual=True)
        self.world = self.simu.world
        self.simu.scene.reparent_to(self.models)

    def add_path(self, u, spline, pathcolor=None):
        if pathcolor is None:
            pathcolor = next(self.colors)
        spl.show_spline2d(
                self.render, spline, u,
                "path_{}".format(self.simu.scene.get_num_children() - 1),
                color=pathcolor)

    def add_domino_run(self, coords, tilt_first_dom=True, color=None):
        self.simu.add_domino_run(coords, tilt_first_dom=tilt_first_dom)
        domino_run_path = self.simu.scene.get_children()[-1]
        # Visual indicators
        if color is None:
            color = next(self.colors)
        for domino in domino_run_path.get_children():
            domino.set_color(Vec4(*color))
        return domino_run_path

    def add_domino_run_from_spline(self, distrib, spline, **kwargs):
        distrib = np.asarray(distrib)
        x, y = spl.splev(distrib, spline)
        head = spl.splang(distrib, spline)
        domino_run_path = self.add_domino_run(
                np.column_stack((x, y, head)), **kwargs)
        return domino_run_path


def show_dominoes_from_coords(coords_lists):
    app = DominoViewer()
    for c in coords_lists:
        app.add_domino_run(c)
    try:
        app.run()
    except SystemExit:
        # Useful if you have to run several in a row.
        app.destroy()


def show_dominoes(distribs, splines):
    app = DominoViewer()
    for d, s in zip(distribs, splines):
        app.add_domino_run_from_spline(d, s)
    try:
        app.run()
    except SystemExit:
        # Useful if you have to run several in a row.
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
