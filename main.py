"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
from functools import partial
from matplotlib import cm
from matplotlib import colors as mcol
import numpy as np
from panda3d.core import NodePath, Point3, Vec3, Vec4, load_prc_file_data

import spline2d as spl
from geom2d import make_rectangle
from primitives import DominoRun, Plane
from uimixins import Drawable, Focusable, Pickerable, Tileable
from uiwidgets import ButtonMenu, DropdownMenu
from viewers import PhysicsViewer
from xp.config import MASS, h, t, w
from xp.domino_predictors import DominoRobustness2

SMOOTHING_FACTOR = .001


class DominoRunMode:
    def __init__(self, parent):
        self.parent = parent

        self.hide_menu_xform = Vec3(0, 0, .2)
        self.menu = ButtonMenu(
                command=self.click_menu,
                items=("DRAW", "GENERATE", "CLEAR", "MOVE DOMINO"),
                text_scale=1,
                text_font=parent.font,
                shadowSize=.2,
                pad=(.2, .2),
                parent=parent.a2dpTopCenter,
                pos=Vec3(-.9*16/9, 0, -.2*9/16) + self.hide_menu_xform,
                scale=.05,
                )

        self.smoothing = SMOOTHING_FACTOR

        self.domrun = None
        self.dompath = None
        self.visual_dompath = None

        self.rob_estimator = DominoRobustness2()

    def click_menu(self, option):
        if option == "DRAW":
            # Clean up the previous orphan drawing if there's one.
            if self.visual_dompath is not None:
                self.visual_dompath.remove_node()
                self.visual_dompath = None
            self.parent.accept_once("mouse1", self.parent.set_draw, [True])
            # Delaying allows to ignore the first "mouse-up"
            # when the menu button is released.
            delayed = partial(
                    self.parent.accept_once, "mouse1-up", self.stop_drawing)
            self.parent.accept_once("mouse1-up", delayed)
        elif option == "GENERATE":
            spline = self.dompath
            # Sample positions
            length = spl.arclength(spline)
            n_doms = int(np.floor(length / (h / 3)))
            u = spl.linspace(spline, n_doms)
            coords = np.column_stack(
                    spl.splev(u, spline) + [spl.splang(u, spline)])
            # Generate run
            run = DominoRun(
                    "domrun_segment_{}".format(self.domrun.get_num_children()),
                    (t, w, h), coords, geom=True, mass=MASS)
            run.create()
            # Add to the scene
            run.attach_to(self.domrun, self.parent.world)
            run.path.set_python_tag('u', u)
            run.path.set_python_tag('spline', spline)
            run.path.set_python_tag('visual_path', self.visual_dompath)
            self.visual_dompath = None
            # Add colors
            self.show_robustness(run.path)
        elif option == "MOVE DOMINO":
            self.set_pickable_dominoes(True)
            self.parent.accept_once("mouse1", self.set_move, [True])
            delayed = partial(
                    self.parent.accept_once, "mouse1-up",
                    self.set_move, [False])
            self.parent.accept_once("mouse1-up", delayed)
        elif option == "CLEAR":
            if self.domrun.get_num_children():
                for domrun_seg in self.domrun.get_children():
                    for domino in domrun_seg.get_children():
                        self.parent.world.remove(domino.node())
                    domrun_seg.get_python_tag('visual_path').remove_node()
                    domrun_seg.remove_node()
            # Clean up the orphan drawing if there's one.
            if self.visual_dompath is not None:
                self.visual_dompath.remove_node()
                self.visual_dompath = None

    def start(self):
        self.parent.enter_design_mode()
        self.show_menu()
        self.domrun = self.parent.models.attach_new_node("domrun")
        self.visual_dompath = None

    def stop(self):
        if self.domrun.get_num_children() == 0:
            self.domrun.remove_node()
        else:
            self.set_pickable_dominoes(False)
        self.domrun = None
        self.dompath = None
        if self.visual_dompath is not None:
            self.visual_dompath.remove_node()
            self.visual_dompath = None
        self.hide_menu()
        self.parent.exit_design_mode()

    def stop_drawing(self):
        self.parent.set_draw(False)
        stroke = self.parent.strokes.pop()
        self.parent.clear_drawing()
        if len(stroke) < 2:
            return
        # Project the drawing
        for point in stroke:
            point[0], point[1], _ = self.parent.mouse_to_ground(point)
        # Smooth the path
        s = self.smoothing
        k = min(3, len(stroke)-1)
        spline = spl.splprep(list(zip(*stroke)), k=k, s=s)[0]
        self.dompath = spline
        # Update visualization
        pencil = self.parent.pencil
        x, y = spl.splev(np.linspace(0, 1, int(1/s)), spline)
        pencil.move_to(x[0], y[0], 0)
        for xi, yi in zip(x, y):
            pencil.draw_to(xi, yi, 0)
        node = pencil.create()
        node.set_name("visual_dompath")
        self.visual_dompath = self.parent.visual.attach_new_node(node)

    def show_robustness(self, domrun_np):
        u = domrun_np.get_python_tag('u')
        spline = domrun_np.get_python_tag('spline')
        coords = np.column_stack(
                spl.splev(u, spline) + [spl.splang(u, spline)])
        scores = np.empty(len(coords))
        scores[1:-1] = self.rob_estimator(coords)
        scores[0] = scores[1]
        scores[-1] = scores[-2]
        self.set_colors(domrun_np, scores)

    def set_colors(self, domrun_np, values, cmap=cm.RdYlGn):
        colors = cmap(values)
        # Increase saturation
        colors[:, :3] = mcol.rgb_to_hsv(colors[:, :3])
        colors[:, 1] *= 2
        colors[:, :3] = mcol.hsv_to_rgb(colors[:, :3])
        # Color the dominoes
        for color, domino in zip(colors, domrun_np.get_children()):
            domino.set_color(Vec4(*color))

    def set_pickable_dominoes(self, pick):
        for domrun_seg in self.domrun.get_children():
            # Extremities are constrained
            for i in range(1, domrun_seg.get_num_children()-1):
                domrun_seg.get_child(i).set_python_tag('pickable', pick)

    def set_move(self, move):
        parent = self.parent
        if move:
            hit_domino = self.parent.get_hit_object()
            if hit_domino is None:
                return
            # We know the mouse in in the window because we have hit_domino.
            self.pos = np.array(parent.mouse_to_ground(
                    parent.mouseWatcherNode.get_mouse()))
            self.moving = hit_domino
            parent.task_mgr.add(self.update_move, "update_move")
        else:
            self.set_pickable_dominoes(False)
            parent.task_mgr.remove("update_move")

    def update_move(self, task):
        if self.parent.mouseWatcherNode.has_mouse():
            new_pos = np.array(self.parent.mouse_to_ground(
                    self.parent.mouseWatcherNode.get_mouse()))
            dom = self.moving
            dom_name = dom.get_name()
            dom_id = int(dom_name[dom_name.rfind('_')+1:])
            domrun_seg = dom.get_parent()
            u = domrun_seg.get_python_tag('u')
            dom_u = u[dom_id]
            spline = domrun_seg.get_python_tag('spline')
            dom_s = spl.arclength(spline, dom_u)
            diff = (new_pos - self.pos)[:2]
            dom_tan = spl.splev(dom_u, spline, 1)
            dom_tan /= np.linalg.norm(dom_tan)
            # Compute new position
            new_dom_s = dom_s + diff.dot(dom_tan)
            new_dom_u = spl.arclength_inv(spline, new_dom_s)
            # Clip (basic)
            new_dom_u = max(new_dom_u, u[dom_id-1])
            new_dom_u = min(new_dom_u, u[dom_id+1])
            # Update
            u[dom_id] = new_dom_u
            new_x, new_y = spl.splev(new_dom_u, spline)
            new_h = spl.splang(new_dom_u, spline)
            dom.set_x(new_x)
            dom.set_y(new_y)
            dom.set_h(new_h)
            # TODO. Update colors
            self.pos = new_pos
        return task.cont

    def hide_menu(self):
        self.menu.posInterval(
                duration=self.parent.menu_anim_time,
                pos=self.menu.get_pos() + self.hide_menu_xform,
                blendType='easeOut'
                ).start()

    def show_menu(self):
        self.menu.posInterval(
                duration=self.parent.menu_anim_time,
                pos=self.menu.get_pos() - self.hide_menu_xform,
                blendType='easeOut'
                ).start()


class MyApp(Tileable, Focusable, Drawable, Pickerable, PhysicsViewer):

    def __init__(self):
        PhysicsViewer.__init__(self)
        Tileable.__init__(self, tile_size=.1)
        Focusable.__init__(self)
        Drawable.__init__(self)
        Pickerable.__init__(self)

        # Initial camera position
        self.min_cam_distance = .2
        self.zoom_speed = .1
        self.cam_distance = 4.
        self.camLens.set_near(.1)
        self.pivot.set_hpr(Vec3(135, 30, 0))

        # Controls
        self.accept("q", self.userExit)
        self.accept("l", self.render.ls)

        # RGM primitives
        floor = Plane("floor", geom=True)
        floor.create()
        floor.attach_to(self.models, self.world)

        self.font = self.loader.load_font("assets/Roboto_regular.ttf")
        bt_shape = NodePath(make_rectangle(4, 2, 0.2, 4))
        bt_shape.set_color(Vec4(65, 105, 225, 255)/255)
        self.add_modes = ("DOMINO RUN", "TODO")
        self.menu = DropdownMenu(
                command=self.click_add_menu,
                items=self.add_modes,
                # Button shape
                relief=None,
                geom=bt_shape,
                # Text
                text="+ ADD",
                text_scale=1,
                text_font=self.font,
                text_fg=Vec4(1, 1, 1, 1),
                # Shadow
                shadowSize=.2,
                # Position and scale
                parent=self.a2dBottomRight,
                pos=Point3(-.3, 0, .25*9/16),
                scale=.04
                )

        self.menu_anim_time = .3
        self.hide_menu_xform = Vec3(.36, 0, 0)

        self.domrun_mode = DominoRunMode(self)

    def click_add_menu(self, option):
        if option == "TODO":
            return
        if option == "DOMINO RUN":
            self.set_show_tile(True)
            self.accept_once("mouse1", self.domrun_mode.start)
            self.accept_once("escape", self.domrun_mode.stop)

    def enter_design_mode(self):
        self.hide_menu()
        self.focus_view(self.tile)

    def exit_design_mode(self):
        self.show_menu()
        self.unfocus_view()
        self.set_show_tile(False)
        self.reset_default_mouse_controls()

    def hide_menu(self):
        self.menu.posInterval(
                duration=self.menu_anim_time,
                pos=self.menu.get_pos() + self.hide_menu_xform,
                blendType='easeOut'
                ).start()

    def show_menu(self):
        self.menu.posInterval(
                duration=self.menu_anim_time,
                pos=self.menu.get_pos() - self.hide_menu_xform,
                blendType='easeOut'
                ).start()


def main():
    load_prc_file_data("", "win-size 1600 900")
    load_prc_file_data("", "window-title Domino Designer")
    load_prc_file_data("", "framebuffer-multisample 1")
    load_prc_file_data("", "multisamples 2")

    app = MyApp()
    app.run()


main()
