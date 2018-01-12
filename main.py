"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
import numpy as np
from panda3d.core import NodePath, Vec3, Vec4, load_prc_file_data

import spline2d as spl
from geom2d import make_rectangle
from primitives import DominoRun, Plane
from uimixins import Drawable, Focusable, Tileable
from uiwidgets import ButtonMenu, DropdownMenu
from viewers import PhysicsViewer
from xp.config import MASS, h, t, w

SMOOTHING_FACTOR = .01


class MyApp(Tileable, Focusable, Drawable, PhysicsViewer):

    def __init__(self):
        PhysicsViewer.__init__(self)
        Tileable.__init__(self)
        Focusable.__init__(self)
        Drawable.__init__(self)

        # Initial camera position
        self.min_cam_distance = .2
        self.zoom_speed = 1.
        self.cam_distance = 30.
        self.pivot.set_hpr(Vec3(135, 30, 0))

        # Controls
        self.accept("q", self.userExit)

        # TODO.
        # Slide in (out) menu when getting in (out of) focus

        # RGM primitives
        floor = Plane("floor", geom=True)
        floor.create()
        floor.attach_to(self.models, self.world)

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
                command=self.click_domino_menu,
                items=("DRAW", "GENERATE", "CLEAR"),
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
        self.accept_once("mouse1", self.focus_on_tile)
        self.accept_once("escape", self.cancel_add_primitive)

    def focus_on_tile(self):
        self.focus_view(self.tile)
        self.menus['domino_run'].show()

    def cancel_add_primitive(self):
        self.ignore("mouse1")
        self.set_show_tile(False)
        self.menus['domino_run'].hide()
        self.unfocus_view()

    def click_domino_menu(self, option):
        if option == "DRAW":
            self.accept("mouse1", self.set_draw, [True])
            # The nested call allows to ignore the first "mouse-up"
            # when the menu button is released.
            self.accept_once(
                    "mouse1-up",
                    lambda: self.accept(
                        "mouse1-up", self.set_draw, [False]))
        elif option == "CLEAR":
            self.clear_drawing()
        elif option == "GENERATE":
            for i, stroke in enumerate(self.strokes):
                # Clean up the points
                stroke = np.array(stroke)
                _, idx = np.unique(
                        stroke.round(decimals=6), return_index=True, axis=0)
                stroke = stroke[np.sort(idx)]  # To get unsorted unique
                # Project onto plane
                for j in range(len(stroke)):
                    stroke[j] = list(self.mouse_to_ground(stroke[j]))[:2]
                # Compute spline
                spline = spl.splprep(np.array(stroke).T, s=SMOOTHING_FACTOR)[0]
                # Sample positions
                length = spl.arclength(spline)
                ndom = int(np.floor(length / (h / 3)))
                u = spl.linspace(spline, ndom)
                coords = np.column_stack(
                        spl.splev(u, spline) + [spl.splang(u, spline)])
                run = DominoRun("domrun_{}".format(i), (t, w, h), coords,
                                geom=True, mass=MASS)
                run.create()
                run.attach_to(self.models, self.world)


def main():
    load_prc_file_data("", "win-size 1600 900")
    load_prc_file_data("", "window-title RGM Designer")
    load_prc_file_data("", "framebuffer-multisample 1")
    load_prc_file_data("", "multisamples 2")

    app = MyApp()
    app.run()


main()
