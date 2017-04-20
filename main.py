#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
import sys
import numpy as np

#from direct.filter.CommonFilters import CommonFilters
from direct.gui.DirectGui import DirectButton
from panda3d.core import Vec3
from panda3d.core import load_prc_file_data
load_prc_file_data("", "win-size 1600 900")
load_prc_file_data("", "window-title RGM Designer")


from primitives import DominoRun #, BallRun, Floor
from uimixins import Tileable, Focusable, Drawable
from viewers import PhysicsViewer


from panda3d.core import Point3
from panda3d.core import LineSegs
from direct.showutil.Rope import Rope


class MyApp(Tileable, Focusable, Drawable, PhysicsViewer):

    def __init__(self):
        PhysicsViewer.__init__(self)
        Tileable.__init__(self)
        Focusable.__init__(self)
        Drawable.__init__(self)

        # Initial camera position
        self.min_cam_distance = .5
        self.cam_distance = 30.
        self.pivot.set_hpr(Vec3(135, 30, 0))

        # Controls
        self.accept("escape", sys.exit)
        self.accept("s", self.make_primitive)

        # TODO.
        # Slide in (out) menu when getting in (out of) focus
        # Smooth the drawing curve
        # Sample it and instantiate dominos

#        self.gui = self.render.attach_new_node("gui")
        font = self.loader.load_font("assets/Roboto_regular.ttf")
        self.add_bt = DirectButton(
                command=self.click_add_button,
                # Background
                relief='flat',
                frameColor=(65/255, 105/255, 225/255, 1),
                pad=(.75, .75),
                # Text
                text="+ ADD",
                text_font=font,
                text_fg=(1, 1, 1, 1),
                # Position and scale
                parent=self.a2dBottomRight,
                pos=Vec3(-.2, 0, .2*9/16),
                scale=.04)

    def click_add_button(self):
        self.set_show_tile(True)
        self.acceptOnce("mouse1", self.focus_on_tile)

    def focus_on_tile(self):
        self.focus_view(self.tile)
        self.acceptOnce("c", self.cancel_add_primitive)
        self.start_drawing()

    def cancel_add_primitive(self):
        self.stop_drawing()
        self.set_show_tile(False)
        self.unfocus_view()

    def start_drawing(self):
        self.accept("mouse1", self.set_draw, [True])
        self.accept("mouse1-up", self.set_draw, [False])

    def stop_drawing(self):
        self.ignore("mouse1")
        self.ignore("mouse1-up")
        self.clear_drawing()

    def make_primitive(self):
        ls = LineSegs(self.strokes)
        ls.create()

        points = []
        for v in ls.get_vertices():
            p = Point3()
            self.mouse_to_ground((v[0], v[2]), p)
            points.append((None, p))

        rope = Rope()
        rope.setup(4, points)
        rope.set_color((1, 0, 0, 1))
        rope.reparent_to(self.render)

        nb_dom = 40
        result = rope.curve.evaluate()
        start = result.getStartT()
        end = result.getEndT()

        pos = []
        head = []
        for t in np.linspace(start, end, nb_dom):
            pt = Point3()
            tan = Vec3()
            result.evalPoint(t, pt)
            result.evalTangent(t, tan)

            pos.append(pt + (0, 0, .2))

            if tan.normalize():
                head.append(Vec3(1, 0, 0).signedAngleDeg(tan, Vec3(0, 0, 1)))
            else:
                head.append(0)

        dominorun_path = self.models.attach_new_node("domino_run")
        dominorun = DominoRun(
                dominorun_path, self.world,
                pos=pos, head=head, extents=(.05, .2, .4), masses=1)
        dominorun.create()

        self.clear_drawing()



def main():
    app = MyApp()
    app.run()


if __name__ == "__main__":
    main()
