"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
from functools import partial
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
        Tileable.__init__(self, tile_size=.1)
        Focusable.__init__(self)
        Drawable.__init__(self)

        # Initial camera position
        self.min_cam_distance = .2
        self.zoom_speed = .1
        self.cam_distance = 4.
        self.camLens.set_near(.1)
        self.pivot.set_hpr(Vec3(135, 30, 0))

        # Controls
        self.accept("q", self.userExit)
        self.accept("l", self.render.ls)

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
                pos=Vec3(-.9*16/9, 0, -.2*9/16),
                scale=.05,
                )
        self.menus['domino_run'].hide()

        self.smoothing = SMOOTHING_FACTOR

    def click_add_menu(self, option):
        if option == "TODO":
            return
        if option == "DOMINO RUN":
            self.set_show_tile(True)
            self.accept_once("mouse1", self.start_domino_design)
            self.accept_once("escape", self.stop_domino_design)

    def start_domino_design(self):
        self.focus_view(self.tile)
        self.menus['domino_run'].show()
        self._tmp_domrun = self.models.attach_new_node("domrun")
        self._tmp_visual_dompath = None

    def stop_domino_design(self):
        if self._tmp_domrun.get_num_children() == 0:
            self._tmp_domrun.remove_node()
        self._tmp_domrun = None
        if self._tmp_visual_dompath is not None:
            self._tmp_visual_dompath.remove_node()
            self._tmp_visual_dompath = None
        self.menus['domino_run'].hide()
        self.unfocus_view()
        self.set_show_tile(False)
        self.reset_default_mouse_controls()

    def stop_drawing(self):
        self.set_draw(False)
        stroke = self.strokes.pop()
        self.clear_drawing()
        if len(stroke) < 2:
            return
        # Project the drawing
        for point in stroke:
            point[0], point[1], _ = self.mouse_to_ground(point)
        # Smooth the path
        s = self.smoothing
        k = min(3, len(stroke)-1)
        spline = spl.splprep(list(zip(*stroke)), k=k, s=s)[0]
        self._tmp_dompath = spline
        # Update visualization
        pencil = self.pencil
        x, y = spl.splev(np.linspace(0, 1, int(1/s)), spline)
        pencil.move_to(x[0], y[0], 0)
        for xi, yi in zip(x, y):
            pencil.draw_to(xi, yi, 0)
        node = pencil.create()
        node.set_name("visual_dompath")
        self._tmp_visual_dompath = self.visual.attach_new_node(node)

    def click_domino_menu(self, option):
        if option == "DRAW":
            # Clean up the previous orphan drawing if there's one.
            if self._tmp_visual_dompath is not None:
                self._tmp_visual_dompath.remove_node()
                self._tmp_visual_dompath = None
            self.accept_once("mouse1", self.set_draw, [True])
            # Delaying allows to ignore the first "mouse-up"
            # when the menu button is released.
            delayed = partial(self.accept_once, "mouse1-up", self.stop_drawing)
            self.accept_once("mouse1-up", delayed)
        elif option == "GENERATE":
            spline = self._tmp_dompath
            # Sample positions
            length = spl.arclength(spline)
            n_doms = int(np.floor(length / (h / 3)))
            u = spl.linspace(spline, n_doms)
            coords = np.column_stack(
                    spl.splev(u, spline) + [spl.splang(u, spline)])
            # Generate run
            run = DominoRun("domrun_segment", (t, w, h), coords,
                            geom=True, mass=MASS)
            run.create()
            # Add to the scene
            run.attach_to(self._tmp_domrun, self.world)
            run.path.set_python_tag('u', u)
            run.path.set_python_tag('spline', spline)
            run.path.set_python_tag('visual_path', self._tmp_visual_dompath)
            self._tmp_visual_dompath = None
        elif option == "CLEAR":
            if self._tmp_domrun.get_num_children():
                for domrun_seg in self._tmp_domrun.get_children():
                    for domino in domrun_seg.get_children():
                        self.world.remove(domino.node())
                    domrun_seg.get_python_tag('visual_path').remove_node()
                    domrun_seg.remove_node()
            # Clean up the orphan drawing if there's one.
            if self._tmp_visual_dompath is not None:
                self._tmp_visual_dompath.remove_node()
                self._tmp_visual_dompath = None


def main():
    load_prc_file_data("", "win-size 1600 900")
    load_prc_file_data("", "window-title Domino Designer")
    load_prc_file_data("", "framebuffer-multisample 1")
    load_prc_file_data("", "multisamples 2")

    app = MyApp()
    app.run()


main()
