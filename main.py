#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
from panda3d.core import load_prc_file_data
from panda3d.core import NodePath
from panda3d.core import Vec3
from panda3d.core import Vec4

from geom2d import make_rectangle
from primitives import DominoMaker
from primitives import Floor
from uimixins import Tileable
from uimixins import Focusable
from uiwidgets import DropdownMenu
from uiwidgets import ButtonMenu
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
        self.accept("q", self.userExit)
        self.accept("a", self.click_add_menu)

        # TODO.
        # Slide in (out) menu when getting in (out of) focus

        # RGM primitives
        floor_path = self.models.attach_new_node("floor")
        floor = Floor(floor_path, self.world)
        floor.create()

#        self.gui = self.render.attach_new_node("gui")
        self.menus = {}
        font = self.loader.load_font("assets/Roboto_regular.ttf")
        bt_shape = NodePath(make_rectangle(4, 2, 0.2, 4))
        bt_shape.set_color(Vec4(65, 105, 225, 255)/255)
        self.add_modes = ("DOMINO RUN", "TODO")
        self.menus['add'] = DropdownMenu(
                command=self.click_add_menu,
                items=self.add_modes,
                # Button shape
                relief=None,
                geom=bt_shape,
                # Text
                text="+ ADD",
                text_scale=1,
                text_font=font,
                text_fg=Vec4(1, 1, 1, 1),
                # Position and scale
                parent=self.a2dBottomRight,
                pos=Vec3(-.3, 0, .25*9/16),
                scale=.04
                )

        self.menus['domino_run'] = ButtonMenu(
                command=lambda option: print(option),
                items=("DRAW", "+1"),
                text_scale=1,
                text_font=font,
                pad=(.2, 0),
                parent=self.a2dpTopCenter,
                pos=Vec3(-.15, 0, -.2*9/16),
                scale=.05,
                )
        self.menus['domino_run'].hide()

    def click_add_menu(self, option):
        if option == "TODO":
            return
        self.set_show_tile(True)
        self.acceptOnce("mouse1", self.focus_on_tile)
        self.acceptOnce("escape", self.cancel_add_primitive)

    def focus_on_tile(self):
        self.focus_view(self.tile)
        self.menus['domino_run'].show()

    def cancel_add_primitive(self):
        self.ignore("mouse1")
        self.set_show_tile(False)
        self.menus['domino_run'].hide()
        self.unfocus_view()


def main():
    load_prc_file_data("", "win-size 1600 900")
    load_prc_file_data("", "window-title RGM Designer")
    load_prc_file_data("", "framebuffer-multisample 1")
    load_prc_file_data("", "multisamples 2")

    app = MyApp()
    app.run()


main()
