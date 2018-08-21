"""
Main GUI.

"""
import os
import sys

import numpy as np
from matplotlib import colors as mcol
from matplotlib import cm
from panda3d.core import NodePath, Point3, Vec3, Vec4, load_prc_file_data

sys.path.insert(0, os.path.abspath('..'))
import core.spline2d as spl  # noqa: E402
from core.primitives import DominoRun, Plane  # noqa: E402
from gui.geom2d import make_rectangle  # noqa: E402
from gui.uimixins import (Drawable, Focusable, Pickerable,  # noqa: E402
                          Tileable)
from gui.uiwidgets import ButtonMenu, DropdownMenu  # noqa: E402
from gui.viewers import PhysicsViewer  # noqa: E402
from xp.config import MASS, TOPPLING_ANGLE, h, t, w  # noqa: E402
from xp.dominoes.creation_global import run_optim  # noqa: E402
from xp.dominoes.geom import tilt_box_forward  # noqa: E402
from xp.dominoes.path import DominoPath  # noqa: E402
from xp.domino_predictors import DominoRobustness2  # noqa: E402

SMOOTHING_FACTOR = .001
PHYSICS_FRAMERATE = 240


class DominoRunMode:
    def __init__(self, parent):
        self.parent = parent
        self.smoothing = SMOOTHING_FACTOR
        self.domrun = None
        self.rob_estimator = DominoRobustness2()
        # Menu
        self.hide_menu_xform = Vec3(0, 0, .2)
        self.menu = ButtonMenu(
                command=self.click_menu,
                items=("CREATE", "REMOVE", "EDIT", "OPTIMIZE"),
                text_scale=1,
                text_font=parent.font,
                shadowSize=.2,
                pad=(.2, .2),
                parent=parent.a2dpTopCenter,
                pos=Vec3(-.9*16/9, 0, -.2*9/16) + self.hide_menu_xform,
                scale=.05,
                )

    def click_menu(self, option):
        if option == "CREATE":
            self.parent.accept_once("mouse1-up", self.set_draw, [True])
        if option == "REMOVE":
            self.parent.accept_once("mouse1-up", self.set_remove, [True])
        elif option == "EDIT":
            self.parent.accept_once("mouse1-up", self.set_move, [True])
        elif option == "OPTIMIZE":
            self.parent.accept_once("mouse1-up", self.set_optimize, [True])

    def start(self):
        self.parent.enter_design_mode()
        self.show_menu()
        self.domrun = self.parent.models.attach_new_node("domrun")

    def stop(self):
        if self.domrun.get_num_children() == 0:
            self.domrun.remove_node()
        self.domrun = None
        self.hide_menu()
        self.parent.exit_design_mode()

    def make_domrun_from_spline(self, spline):
        # Sample positions
        length = spl.arclength(spline)
        n_doms = int(np.floor(length / (h / 3)))
        u = spl.linspace(spline, n_doms)
        coords = np.column_stack(
                spl.splev(u, spline) + [spl.splang(u, spline)])
        # Generate run
        run = DominoRun(
                "domrun_segment_{}".format(self.domrun.get_num_children()),
                (t, w, h), coords, geom='LD', mass=MASS)
        run.create()
        # Tilt first domino
        tilt_box_forward(run.path.get_child(0), TOPPLING_ANGLE + 1)
        # Add to the scene
        run.attach_to(self.domrun, self.parent.world)
        self.parent._create_cache()
        # Add visual path
        pencil = self.parent.pencil
        x, y = spl.splev(np.linspace(0, 1, int(1/self.smoothing)), spline)
        pencil.move_to(x[0], y[0], 0)
        for xi, yi in zip(x, y):
            pencil.draw_to(xi, yi, 0)
        node = pencil.create()
        node.set_name("visual_dompath")
        visual_dompath = self.parent.visual.attach_new_node(node)
        # Set tags
        run.path.set_python_tag('u', u)
        run.path.set_python_tag('length', length)
        run.path.set_python_tag('spline', spline)
        run.path.set_python_tag('visual_dompath', visual_dompath)
        for child in run.path.get_children():
            child.set_python_tag('pickable', True)
        # Add colors
        self.show_robustness(run.path)

    def show_robustness(self, domrun_np):
        u = domrun_np.get_python_tag('u')
        spline = domrun_np.get_python_tag('spline')
        coords = np.column_stack(
                spl.splev(u, spline) + [spl.splang(u, spline)])
        scores = np.empty(len(coords))
        scores[2:] = self.rob_estimator(coords)
        scores[:2] = scores[2]
        self.set_colors(domrun_np, scores)

    def set_colors(self, domrun_np, values, cmap=cm.autumn):
        colors = cmap(values)
        # Increase saturation
        colors[:, :3] = mcol.rgb_to_hsv(colors[:, :3])
        colors[:, 1] *= 2
        colors[:, :3] = mcol.hsv_to_rgb(colors[:, :3])
        # Color the dominoes
        for color, domino in zip(colors, domrun_np.get_children()):
            domino.set_color(Vec4(*color))

    def set_draw(self, draw):
        parent = self.parent
        if draw:
            parent.accept_once("mouse1", parent.set_draw, [True])
            parent.accept_once("mouse1-up", self.set_draw, [False])
        else:
            parent.set_draw(False)
            stroke = parent.strokes.pop()
            parent.clear_drawing()
            if len(stroke) < 2:
                return
            # Project the drawing
            for point in stroke:
                point[0], point[1], _ = parent.mouse_to_ground(point)
            # Smooth the path
            s = self.smoothing
            k = min(3, len(stroke)-1)
            spline = spl.splprep(list(zip(*stroke)), k=k, s=s)[0]
            self.make_domrun_from_spline(spline)

    def set_highlight(self, highlight):
        parent = self.parent
        if highlight:
            self.highlighted = None
            parent.task_mgr.add(self.update_highlight, "update_highlight")
        else:
            parent.task_mgr.remove("update_highlight")
            if self.highlighted is not None:
                self.highlighted.clear_render_mode()
                self.highlighted = None

    def update_highlight(self, task):
        parent = self.parent
        hit = parent.get_hit_object()
        if self.highlighted is None:
            if hit is None:
                pass
            else:
                self.highlighted = hit
                self.highlighted.set_render_mode_filled_wireframe(1)
        else:
            if hit is None:
                self.highlighted.clear_render_mode()
                self.highlighted = None
            else:
                if hit == self.highlighted:
                    pass
                else:
                    self.highlighted.clear_render_mode()
                    self.highlighted = hit
                    self.highlighted.set_render_mode_filled_wireframe(1)
        return task.cont

    def set_remove(self, remove):
        parent = self.parent
        if remove:
            parent.pick_level = 1
            self.set_highlight(True)
            parent.accept_once("mouse1", self.remove_selected)
            parent.accept_once("mouse1-up", self.set_remove, [False])
        else:
            self.set_highlight(False)

    def remove_selected(self):
        parent = self.parent
        domrun_seg = parent.get_hit_object()
        if domrun_seg is None:
            return
        for domino in domrun_seg.get_children():
            parent.world.remove(domino.node())
        parent._create_cache()
        domrun_seg.get_python_tag('visual_dompath').remove_node()
        domrun_seg.remove_node()

    def set_move(self, move):
        parent = self.parent
        if move:
            parent.pick_level = 0
            self.set_highlight(True)
            parent.accept_once("mouse1", self.move_selected)
            parent.accept_once("mouse1-up", self.set_move, [False])
        else:
            self.set_highlight(False)
            parent.task_mgr.remove("update_move")
            parent._create_cache()

    def move_selected(self):
        parent = self.parent
        hit_domino = parent.get_hit_object()
        if hit_domino is None:
            return
        domrun_seg = hit_domino.get_parent()
        first = domrun_seg.get_child(0)
        last = domrun_seg.get_child(domrun_seg.get_num_children() - 1)
        if hit_domino in (first, last):
            return
        # We know the mouse in in the window because we have hit_domino.
        self.pos = np.array(parent.mouse_to_ground(
                parent.mouseWatcherNode.get_mouse()))
        self.moving = hit_domino
        parent.task_mgr.remove("update_highlight")
        parent.task_mgr.add(self.update_move, "update_move")

    def update_move(self, task):
        parent = self.parent
        if not parent.mouseWatcherNode.has_mouse():
            return task.cont
        new_pos = np.array(parent.mouse_to_ground(
                parent.mouseWatcherNode.get_mouse()))
        dom = self.moving
        dom_name = dom.get_name()
        dom_id = int(dom_name[dom_name.rfind('_')+1:])
        domrun_seg = dom.get_parent()
        u = domrun_seg.get_python_tag('u')
        dom_u = u[dom_id]
        length = domrun_seg.get_python_tag('length')
        spline = domrun_seg.get_python_tag('spline')
        diff = (new_pos - self.pos)[:2]
        dom_tan = spl.splev(dom_u, spline, 1)
        dom_tan /= np.linalg.norm(dom_tan)
        # Compute new position
        new_dom_u = dom_u + diff.dot(dom_tan) / length
        # Clip (basic)
        new_dom_u = max(new_dom_u, u[dom_id-1])
        new_dom_u = min(new_dom_u, u[dom_id+1])
        # Clip (advanced)
        old_pos = dom.get_pos()
        old_h = dom.get_h()
        new_x, new_y = spl.splev(new_dom_u, spline)
        new_h = spl.splang(new_dom_u, spline)
        dom.set_pos(new_x, new_y, old_pos.z)
        dom.set_h(new_h)
        dom_bef = domrun_seg.get_child(dom_id-1).node()
        dom_aft = domrun_seg.get_child(dom_id+1).node()
        dom_node = dom.node()
        test_bef = parent.world.contact_test_pair(dom_node, dom_bef)
        test_aft = parent.world.contact_test_pair(dom_node, dom_aft)
        if test_bef.get_num_contacts() + test_aft.get_num_contacts() > 0:
            dom.set_pos(old_pos)
            dom.set_h(old_h)
            return task.cont
        # Update
        u[dom_id] = new_dom_u
        self.show_robustness(domrun_seg)
        self.pos = new_pos

        return task.cont

    def set_optimize(self, optimize):
        parent = self.parent
        if optimize:
            parent.pick_level = 1
            self.set_highlight(True)
            parent.accept_once("mouse1", self.optimize_selected)
            parent.accept_once("mouse1-up", self.set_optimize, [False])
        else:
            self.set_highlight(False)

    def optimize_selected(self):
        parent = self.parent
        domrun_seg = parent.get_hit_object()
        if domrun_seg is None:
            return
        u = domrun_seg.get_python_tag('u')
        spline = domrun_seg.get_python_tag('spline')
        init_doms = DominoPath(u, spline)
        best_doms = run_optim(init_doms, self.rob_estimator, method='minimize')
        # Update
        for domino, (x, y, a) in zip(
                domrun_seg.get_children(), best_doms.coords):
            domino.set_pos(x, y, domino.get_z())
            domino.set_h(a)
        domrun_seg.set_python_tag('u', best_doms.u)
        print("old u", u, "new u", best_doms.u)
        self.show_robustness(domrun_seg)
        parent._create_cache()

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
        PhysicsViewer.__init__(self, PHYSICS_FRAMERATE)
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
        floor = Plane("floor", geom='LD')
        floor.create()
        floor.attach_to(self.models, self.world)

        self.font = self.loader.load_font("../../assets/Roboto_regular.ttf")
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
