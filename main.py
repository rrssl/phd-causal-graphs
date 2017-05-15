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
from scipy.optimize import minimize, minimize_scalar
from shapely.affinity import rotate, translate
from shapely.geometry import box

#from direct.filter.CommonFilters import CommonFilters
from direct.gui.DirectGui import DirectButton
from panda3d.core import Vec3, Vec4, Mat3
from panda3d.core import load_prc_file_data
load_prc_file_data("", "win-size 1600 900")
load_prc_file_data("", "window-title RGM Designer")
from panda3d.core import Point3
from panda3d.core import LineSegs

from primitives import DominoMaker, Floor #, BallRun
from uimixins import Tileable, Focusable, Drawable
from viewers import PhysicsViewer


def arclength(tck):
    # In Cartesian coords, s = integral( sqrt(x'**2 + y'**2) )
    def speed(u):
        dx, dy = splev(u, tck, 1)
        return np.sqrt(dx*dx + dy*dy)
    return quad(speed, 0, 1)[0]

def remove_all_children(node, world):
    rmw = world.remove
    rmc = node.remove_child
    for n in node.get_children(): rmw(n); rmc(n)


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
        self.accept("s", self.make_domino_run)

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

    ### Domino-specific
    @staticmethod
    def test_valid_path(tck, extents):
        """Test if the domino path is geometrically valid.

        Supposing constant domino extents (w,h,t), a path is valid
        iff a circle of radius w, swept along the path, never
        intersects the path more than twice (once 'in', once 'out').

        """
        return True

    @staticmethod
    def test_valid_run(run_np, angvel_init=None):
        """Test if the domino run is physically valid.

        A layout is physically valid iff:
            - No domino initially intersects another one.
            - All dominoes topple in the order implicitly defined by 'pos'.

        """
        first_domino = run_np.get_child(0)
        if angvel_init is not None:
            first_domino.node().set_angular_velocity(angvel_init)
        return True

    def place_dominoes(self, tck, extents, angvel_init):
        """Place the dominoes along the path.

        Returns
        -------
        pos : (n,3) floats
            Positions of the center of each domino.
        head : (n,) floats
            Heading of each domino.

        """
        world = self.world
        reset_physics = self.reset_physics
        _create_cache = self._create_cache

        if not self.test_valid_path(tck, extents):
            print("Invalid path!", file=sys.stderr)
            return
        # Attach a test node to the scene and initialize the domino factory.
        run_np = self.models.attach_new_node("domino_run_temp")
        domino_factory = DominoMaker(run_np, world)

        # First try a shortcut: distribute all dominoes linearly
        # (wrt the parameter).
        thickness, width, height = extents
        length = arclength(tck)
        nb_dom = int(length / (3 * thickness))
        u = np.linspace(0, 1, nb_dom)
        # Utility function
        def get_pos_head(u):
            pos = np.column_stack(splev(u, tck) + [np.full(len(u), height*.5)])
            head = np.degrees(np.arctan2(*splev(u, tck, 1)[::-1]))
            return pos, head
        pos, head = get_pos_head(u)
        # Create the domino run
        for i, (p, h) in enumerate(zip(pos, head)):
            domino_factory.add_domino(
                    Vec3(*p), h, extents, mass=1, prefix="d_{}".format(i))
        # Return coords if valid
        if self.test_valid_run(run_np, angvel_init):
            remove_all_children(run_np.node(), world)
            run_np.remove_node()
            return pos, head
        else:
            remove_all_children(run_np.node(), world)

        # Now try the regular method.
        # Define constants.
        max_nb_dom = int(length / thickness)
        timestep = 1 / 60
        max_time = 1.
        # Initialize variables.
        u = [0.]
        doms = []
        pos, head = get_pos_head(u[-1])
        doms.append(domino_factory.add_domino(
                Vec3(*pos), head, extents, mass=1, prefix="d_0"))
        doms[0].node().set_angular_velocity(angvel_init)
        _create_cache()
        last_ustep = 0.
        i = 1
        while (1. - u[-1] > last_ustep) and (len(u) < max_nb_dom):
            # Compute the lower bound.
            dx, dy = splev(u[-1], tck, 1)
            ds = np.sqrt(dx*dx + dy*dy)
            ui = u[-1] + thickness / ds # first-order approximation
            # Initialize the new domino.
            pos, head = get_pos_head(ui)
            doms.append(domino_factory.add_domino(
                    Vec3(*pos), head, extents, mass=1,
                    prefix="d_{}".format(i)))
            new_np = doms[-1]
            new = new_np.node()
            base = box(-thickness*.5, -width*.5,
                        thickness*.5,  width*.5)
            last_base = translate(rotate(base, doms[-2].get_h()),
                                  doms[-2].get_x(), doms[-2].get_y())
            last = doms[-2].node()
            def objective(ui):
                # Reset simulation.
                reset_physics()
                new_np.clear_transform()
                new.clear_forces()
                new.set_linear_velocity(0)
                new.set_angular_velocity(0)
                new.set_active(True)
                # Put the new domino at its new position.
                pos, head = get_pos_head(ui)
                new_np.set_pos(Vec3(*p))
                new_np.set_h(h)
                # Make sure the new and last dominoes are not intersecting.
                new_base = translate(rotate(base, head), pos[0], pos[1])
                c = last_base.intersection(new_base).area
                if c > 0:
                    return c
                # Run the simulation until the impact, or until time is up.
                t = 0.
                test = world.contact_test_pair(last, new)
                while test.get_num_contacts() == 0 and t < max_time:
                    t += timestep
                    world.do_physics(timestep)
                    test = world.contact_test_pair(last, new)
                angvel = new.get_angular_velocity()
                angvel = Mat3.rotate_mat(-new_np.get_h()).xform(angvel)
                return -angvel[1]

            res = minimize_scalar(objective, bounds=(ui, 1.), method='bounded')
            if res.fun[-1] > 0.:
                print("Non-negative objective!\n", res.fun)
            ui = res.x

            print("Placing domino no. {} at u = {}".format(len(u + 1), ui))
            u.append(ui)
            last_ustep = ui[-1] - u[-2]
            i += 1
            _create_cache()

        pos, head = get_pos_head(u)
        if self.test_valid_run(pos, head, extents, angvel_init):
            remove_all_children(run_np.node(), world)
            run_np.remove_node()
            self._physics_cache.clear()
            return pos, head

#        nb_dom = len(u_init)
#        def rectangle(u, return_coord=False):
#            base = box(-extents[0]/2, -extents[1]/2,
#                        extents[0]/2,  extents[1]/2)
#            x, y = splev(u, tck)
#            ang = np.arctan2(*splev(u, tck, 1)[::-1])
#            rects = [translate(rotate(base, ai, use_radians=True), xi, yi)
#                     for xi, yi, ai in zip(x, y, ang)]
#            return rects, (x, y, ang) if return_coord else rects
#        r_init, c_init = rectangle(u_init, return_coord=True)
#
#        def cached_rectangle(u, *args, _cache={}, **kwargs):
#            try:
#                return _cache[tuple(u)]
#            except KeyError:
#                _cache[tuple(u)] = rectangle(u, *args, **kwargs)
#                return _cache[tuple(u)]
#        def nonoverlap_cst(u):
#            r = cached_rectangle(u, True)[0]
#            r = r_init[:1] + r + r_init[-1:]
#            return np.array([r[i].intersection(r[i+1]).area
#                             for i in range(nb_dom-2)])
#        def rectangle2(u):
#            base = box(-extents[2]/2, -extents[1]/2,
#                        extents[2]/2,  extents[1]/2)
#            _, (x, y, ang) = cached_rectangle(u, True)
#            rects = [
#                rotate(translate(base, xi+extents[2]/2, yi), ai,
#                       use_radians=True)
#                for xi, yi, ai in zip(x, y, ang)]
#            return rects
#        def cached_rectangle2(u, _cache={}):
#            try:
#                return _cache[tuple(u)]
#            except KeyError:
#                _cache[tuple(u)] = rectangle2(u)
#                return _cache[tuple(u)]
#        def reach_cst(u):
#            r = r_init[:1] + cached_rectangle2(u) + r_init[-1:]
#            return np.array([r[i].intersection(r[i+1]).area
#                             for i in range(nb_dom-2)])
#        def overlap_energy(r1, r2):
#            return np.exp(r1.intersection(r2).area - min(r1.area, r2.area))
#        def lennard_jones(t1, t2, deq=3*extents[0]):
#            d = quad(speed, t1, t2)[0] + 1e-12
#            return .5 * ((deq / d)**12 - 2 * (deq / d)**6)
#        def objective(u):
#            rects = r_init[:1] + rectangle(u) + r_init[-1:]
##            u = [0.] + list(u) + [1.]
##            avg_dist = sum(
##                    ri.centroid.distance(rj.centroid)
##                    for ri, rj in zip(rects[:-1], rects[1:])
##                    ) / len(rects)
#            return sum(
#                    overlap_energy(ri, rj)
##                    + abs(avg_dist - ri.centroid.distance(rj.centroid))
##                    / avg_dist
##                    + lennard_jones(u[i], u[i+1])
#                    for ri, rj in zip(rects[:-1], rects[1:])
#                    )
#        def objective(u):
#            _, coords = cached_rectangle(u, return_coord=True)
#            ang = np.hstack([c_init[2][:1], coords[2], c_init[2][-1:]])
#            return abs(ang[1:] - ang[:-1]).sum()
#        cst = [
#            dict(type='eq', fun=nonoverlap_cst),
#            dict(type='ineq', fun=reach_cst),
#            ]
#        res = minimize(objective, u_init[1:-1], method='SLSQP',
#                       bounds=[(0.,1.)]*(len(u_init)-2), constraints=cst,
#                       options=dict(disp=True))
#        return np.hstack([0., res.x, 1.])

    def make_domino_run(self, extents=Vec3(.05,.2,.4)):
        # Prepare the points for smoothing.
        ls = LineSegs(self.strokes)
        ls.create()
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
        u = np.linspace(0., 1., 100)
        new_vertices = np.column_stack(splev(u, tck) + [np.zeros(len(u))])
        ls = LineSegs("smoothed")
        ls.set_thickness(4)
        ls.set_color((1, 1, 0, 1))
        ls.move_to(*new_vertices[0])
        for v in new_vertices[1:]:
            ls.draw_to(*v)
        self.render.attach_new_node(ls.create())

        # Determine position and orientation of each domino.
        angvel_init = Mat3.rotate_mat(
                np.degrees(np.arctan2(*splev(0, tck, 1)[::-1]))
                ).xform(Vec3(0., 10., 0.))
        pos, head = self.place_dominoes(tck, extents, angvel_init)

        # Instantiate dominoes.
        domino_run_np = self.models.attach_new_node("domino_run")
        domino_factory = DominoMaker(domino_run_np, self.world)
        for i, (p, h) in enumerate(zip(pos, head)):
            domino_factory.add_domino(
                    Vec3(*p), h, extents, mass=1, prefix="domino_{}".format(i))

        first_domino = domino_run_np.get_child(0)
        first_domino.node().set_angular_velocity(angvel_init)

        self._create_cache()

    def show_rects(self, rects, label="rectangles"):
        ls = LineSegs(label)
        ls.set_thickness(2)
        ls.set_color((1, 0, 0, 1))
        for r in rects:
            vert = list(r.exterior.coords)
            ls.move_to(vert[0][0], vert[0][1], 0.)
            for v in vert[1:]:
                ls.draw_to(v[0], v[1], 0.)
            self.render.attach_new_node(ls.create())

def main():
    app = MyApp()
    app.run()


if __name__ == "__main__":
    main()
