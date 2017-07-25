#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
from direct.gui.DirectGui import DirectButton
from direct.gui.DirectGui import DirectOptionMenu
from panda3d.core import load_prc_file_data
from panda3d.core import Vec3
from panda3d.core import Vec4

from primitives import DominoMaker
from primitives import Floor
from uimixins import Tileable
from uimixins import Focusable
from viewers import PhysicsViewer


class MyApp(Tileable, Focusable, PhysicsViewer):

    def __init__(self):
        PhysicsViewer.__init__(self)
        Tileable.__init__(self)
        Focusable.__init__(self)

        # Initial camera position
        self.min_cam_distance = .2
        self.zoom_speed = 1.
        self.cam_distance = 30.
        self.pivot.set_hpr(Vec3(135, 30, 0))

        # Controls
        self.accept("escape", self.userExit)
        self.accept("a", self.click_add_button)

        # TODO.
        # Slide in (out) menu when getting in (out of) focus

        # RGM primitives
        floor_path = self.models.attach_new_node("floor")
        floor = Floor(floor_path, self.world)
        floor.create()

#        self.gui = self.render.attach_new_node("gui")
        font = self.loader.load_font("assets/Roboto_regular.ttf")
        self.add_bt = DirectButton(
                command=self.click_add_button,
                # Background
                relief='flat',
                frameColor=Vec4(65, 105, 225, 255)/255,
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

    def click_add_menu(self, option):
        pass

    def focus_on_tile(self):
        self.focus_view(self.tile)
        self.acceptOnce("c", self.cancel_add_primitive)

    def cancel_add_primitive(self):
        self.stop_drawing()
        self.set_show_tile(False)
        self.unfocus_view()


def main():
    load_prc_file_data("", "win-size 1600 900")
    load_prc_file_data("", "window-title RGM Designer")

    app = MyApp()
    app.run()


main()
