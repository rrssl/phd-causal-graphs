#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
import sys
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from shapely.affinity import rotate, translate
from shapely.geometry import box

#from direct.filter.CommonFilters import CommonFilters
from direct.gui.DirectGui import DirectButton
from panda3d.core import Vec3, Vec4
from panda3d.core import load_prc_file_data
load_prc_file_data("", "win-size 1600 900")
load_prc_file_data("", "window-title RGM Designer")
from panda3d.core import Point3
from panda3d.core import LineSegs

from primitives import DominoRun, Floor #, BallRun
from uimixins import Tileable, Focusable, Drawable
from viewers import PhysicsViewer


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
        self.accept("escape", sys.exit)
        self.accept("a", self.click_add_button)
        self.accept("s", self.make_primitive)

        # TODO.
        # Slide in (out) menu when getting in (out of) focus
        # Sample it and instantiate dominoes

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
        # Prepare the points for smoothing.
        points = []
        mouse_to_ground = self.mouse_to_ground
        vertices = ls.get_vertices()
        last_added = None
        for v in vertices:
            # Ensure that no two consecutive points are duplicates.
            # Alternatively we could do before the loop:
            # vertices = [v[0] for v in itertools.groupby(vertices)]
            if v != last_added:
                last_added = v

                p = Point3()
                mouse_to_ground((v[0], v[2]), p)
                points.append((p[0], p[1]))
        # Smooth the trajectory.
        # scipy's splprep is more convenient here than panda3d's Rope class,
        # since it gives better control over curve smoothing.
        tck, _ = splprep(np.array(points).T, s=.1)

        # Remove the drawing...
        self.clear_drawing()
        # ... and show the smoothed curve.
        t = np.linspace(0., 1., 100)
        new_vertices = np.column_stack(splev(t, tck) + [np.zeros(len(t))])
        ls = LineSegs("smoothed")
        ls.set_thickness(4)
        ls.set_color((1, 1, 0, 1))
        ls.move_to(*new_vertices[0])
        for v in new_vertices[1:]:
            ls.draw_to(*v)
        self.render.attach_new_node(ls.create())

        # Compute the arc length: in Cartesian coords,
        # s = integral( sqrt(x'**2 + y'**2) )
        def speed(t):
            dx, dy = splev(t, tck, 1)
            return np.sqrt(dx*dx + dy*dy)
        length = quad(speed, 0, 1)[0]
        # Determine position and orientation of each domino.
        extents = (.05, .2, .4)
        nb_dom = int(length / (2.5 * extents[0]))

        def rectangle(t):
            base = box(-extents[0]/2, -extents[1]/2,
                        extents[0]/2,  extents[1]/2)
            x, y = splev(t, tck)
            ang = np.arctan2(*splev(t, tck, 1)[::-1])
            return [translate(rotate(base, ai, use_radians=True), xi, yi)
                    for xi, yi, ai in zip(x, y, ang)]
        t_init = np.linspace(0., 1., nb_dom)
        r_init = rectangle(t_init)

        # Show initial domino positions
        ls = LineSegs("initial domino position")
        ls.set_thickness(2)
        ls.set_color((1, 0, 0, 1))
        for r in r_init:
            vert = list(r.exterior.coords)
            ls.move_to(vert[0][0], vert[0][1], 0.)
            for v in vert[1:]:
                ls.draw_to(v[0], v[1], 0.)
            self.render.attach_new_node(ls.create())

        def overlap_energy(r1, r2):
            return np.exp(r1.intersection(r2).area - min(r1.area, r2.area))
        def lennard_jones(t1, t2, deq=3*extents[0]):
            d = quad(speed, t1, t2)[0] + 1e-12
            return .5 * ((deq / d)**12 - 2 * (deq / d)**6)
        def objective(t):
            rects = r_init[:1] + rectangle(t) + r_init[-1:]
#            t = [0.] + list(t) + [1.]
#            avg_dist = sum(
#                    ri.centroid.distance(rj.centroid)
#                    for ri, rj in zip(rects[:-1], rects[1:])
#                    ) / len(rects)
            return sum(
                    overlap_energy(ri, rj)
#                    + abs(avg_dist - ri.centroid.distance(rj.centroid))
#                    / avg_dist
#                    + lennard_jones(t[i], t[i+1])
                    for ri, rj in zip(rects[:-1], rects[1:])
                    )
        res = minimize(objective, t_init[1:-1],
                       method='L-BFGS-B', bounds=[(0.,1.)]*(len(t_init)-2),
                       options=dict(disp=True))
        t = np.hstack([0, res.x, 1])

#        t = t_init
        pos = np.column_stack(splev(t, tck) + [np.full(nb_dom, .2)])
        head = np.arctan2(*splev(t, tck, 1)[::-1]) * 180 / np.pi
        # Instantiate dominoes.
        domino_run_np = self.models.attach_new_node("domino_run")
        domino_run = DominoRun(
                domino_run_np, self.world,
                pos=pos, head=head, extents=extents, masses=1)
        domino_run.create()

        first_domino = domino_run_np.find("domino_0_solid")
        first_domino.set_r(15)

        self._create_cache()

def main():
    app = MyApp()
    app.run()


if __name__ == "__main__":
    main()
