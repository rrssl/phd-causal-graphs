#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loading RGM primitives, showing motion paths.

@author: Robin Roussel
"""
import numpy as np
from panda3d.core import Vec3
from panda3d.core import load_prc_file_data
load_prc_file_data("", "win-size 1024 768")
load_prc_file_data("", "window-title RGM Designer")

from primitives import BallRun, Floor, DominoRun
from viewers import FutureViewer


class MyApp(FutureViewer):

    def __init__(self):
        super().__init__()

        # Initial camera position
        self.cam_distance = 10.
        self.pivot.set_h(self.pivot, -15)
        self.pivot.set_p(self.pivot, 15)

        # RGM primitives
        floor_path = self.models.attach_new_node("floor")
        floor = Floor(floor_path, self.world)
        floor.create()

        ballrun_path = self.models.attach_new_node("ball_run")
        params = {
            'ball_pos': Vec3(-1., 0., 1.),
            'ball_rad': .1,
            'ball_mass': .2,
            'block_pos': Vec3(-.5, 0., .5),
            'block_hpr': Vec3(0., 0., 1.),
            'block_ext': Vec3(2., .5, .1)}
        ballrun = BallRun(ballrun_path, self.world, **params)
        ballrun.create()

        dominorun_path = self.models.attach_new_node("domino_run")
        nb_dom = 10
        t = np.linspace(0, -np.pi/2, nb_dom)
        pos = np.vstack([np.cos(t), np.sin(t), np.full(nb_dom, .2)]).T
        pos += [1, 1, 0]
        params = {
#            'pos': np.vstack([np.arange(nb_dom) * .2 + 1.,
#                              np.zeros(nb_dom),
#                              np.full(nb_dom, .2)]).T,
#            'head': np.zeros(nb_dom),
            'pos': pos,
            'head': -t[::-1] * 180 / np.pi,
            'extents': np.tile([.05, .2, .4], (nb_dom, 1)),
            'masses': np.ones(nb_dom)}
        dominorun = DominoRun(dominorun_path, self.world, **params)
        dominorun.create()

        # Controls
        self.rotate_step = 1.
        self.acceptOnce("arrow_up", self.rotate_block, [True])
        self.acceptOnce("arrow_down", self.rotate_block, [False])

    def rotate_block(self, direct):
        path = self.models.find("**/block_solid")
        if direct:
            path.set_r(path.get_r() + self.rotate_step)
            # Reaccept up key
            self.acceptOnce("arrow_up", self.rotate_block, [True])
        else:
            path.set_r(path.get_r() - self.rotate_step)
            # Reaccept down key
            self.acceptOnce("arrow_down", self.rotate_block, [False])
        # Update future
        # Note: no need to update cache if only static objects are moved.
        # self._create_cache()
        self.update_future()


def main():
    app = MyApp()
    app.run()


if __name__ == "__main__":
    main()
