"""
Drawing a path and sampling dominoes along it.

"""
import sys

from direct.gui.DirectGui import DirectButton
import numpy as np
from panda3d.bullet import BulletBodyNode
from panda3d.core import LineSegs
from panda3d.core import load_prc_file_data
from panda3d.core import Mat3
from panda3d.core import Point3
from panda3d.core import Vec3
from panda3d.core import Vec4
from scipy.interpolate import splev
import scipy.optimize as opt
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box
from shapely.geometry import Point
from shapely.geometry import LineString

from primitives import DominoMaker
from primitives import Floor
import spline2d as spl
import uimixins as ui
from viewers import PhysicsViewer


def remove_all_bullet_children(node, bt_world):
    rmw = bt_world.remove
    rmc = node.remove_child
    for ni in node.get_children():
        if ni.is_of_type(BulletBodyNode):
            rmw(ni)
            rmc(ni)


def show_rects(parent, rects, label="rectangles"):
    """Create a LineSegs instance representing a sequence of Shapely boxes."""
    ls = LineSegs(label)
    ls.set_thickness(2)
    ls.set_color((1, 0, 0, 1))
    for r in rects:
        vert = list(r.exterior.coords)
        ls.move_to(vert[0][0], vert[0][1], 0.)
        for v in vert[1:]:
            ls.draw_to(v[0], v[1], 0.)
        parent.attach_new_node(ls.create())


class InvalidPathError(Exception):
    pass


class DominoPathDrawer(ui.Tileable, ui.Focusable, ui.Drawable, PhysicsViewer):

    def __init__(self):
        PhysicsViewer.__init__(self)
        ui.Tileable.__init__(self)
        ui.Focusable.__init__(self)
        ui.Drawable.__init__(self)

        # Initial camera position
        self.min_cam_distance = .2
        self.zoom_speed = 1.
        self.cam_distance = 30.
        self.pivot.set_hpr(Vec3(135, 30, 0))

        # Controls
        self.accept("escape", sys.exit)
        self.accept("a", self.click_add_button)
        self.accept("s", self.make_domino_run)
        self.accept("f", self.save_drawing)

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

    # Domino-specific
    @staticmethod
    def test_valid_path(tck, extents):
        """Test if the domino path is geometrically valid.

        Supposing constant domino extents (w,h,t), a path is valid
        iff a circle of radius w, swept along the path, never
        intersects the path more than twice (once 'in', once 'out').

        """
        l = spl.arclength(tck)
        t, w, _ = extents
        u = np.linspace(0, 1, int(l/t))
        vertices = np.column_stack(splev(u, tck))
        path = LineString(vertices)
        base_circle = Point(0, 0).buffer(w)
        for x, y in vertices:
            circle = translate(base_circle, x, y)
            try:
                if len(circle.boundary.intersection(path)) > 2:
                    return False
            except TypeError:  # Happens when intersection is a single point.
                pass
        return True

    @staticmethod
    def get_valid_path_ranges(tck, extents):
        """Get a list of valid domino path segments.

        For the definition of validity see test_valid_path.

        """
        l = spl.arclength(tck)
        t, w, _ = extents
        u = np.linspace(0, 1, int(l/t))
        vertices = np.column_stack(splev(u, tck))
        path = LineString(vertices)
        base_circle = Point(0, 0).buffer(w)
        valid = []
        rng_start = rng_end = 0.
        opened = False
        for uj, (x, y) in zip(u, vertices):
            circle = translate(base_circle, x, y)
            try:
                n = len(circle.boundary.intersection(path))
            except TypeError:  # Happens when intersection is a single point.
                n = 1
            if n < 3:  # Current vertex is valid.
                if opened:
                    rng_end = uj
                else:
                    opened = True
                    rng_start = rng_end = uj
            else:  # Current vertex is not valid.
                if opened:
                    valid.append((rng_start, rng_end))
                opened = False
        else:
            if opened:
                valid.append((rng_start, rng_end))
        return valid

    @staticmethod
    def test_valid_run(run_np, angvel_init=None):
        """Test if the domino run is physically valid.

        A layout is physically valid iff:
            - No domino initially intersects another one.
            - All dominoes topple in the order implicitly defined by 'pos'.

        """
#        first_domino = run_np.get_child(0)
#        if angvel_init is not None:
#            first_domino.node().set_angular_velocity(angvel_init)
        return True

    def place_dominoes_(self, tck, extents):
        """Place the dominoes along the path.

        Returns
        -------
        pos : (n,3) floats
            Positions of the center of each domino.
        head : (n,) floats
            Heading of each domino.

        """
        if not self.test_valid_path(tck, extents):
            raise InvalidPathError
        thickness, width, height = extents
        length = spl.arclength(tck)
        nb_dom = int(length / (3 * thickness))
        u = np.linspace(0, 1, nb_dom)
        pos = spl.splev3d(u, tck, .5*height)
        head = spl.splang(u, tck)
        # Create the domino run
        return pos, head

    def place_dominoes(self, tck, extents, angvel_init, skip_uniform=True):
        """Place the dominoes along the path.

        Returns
        -------
        pos : (n,3) floats
            Positions of the center of each domino.
        head : (n,) floats
            Heading of each domino.

        """
        if not self.test_valid_path(tck, extents):
            raise InvalidPathError
        # Make some attributes and methods local for faster access.
        world = self.world
        reset_physics = self.reset_physics
        _add_to_cache = self._add_to_cache
        # Attach a test node to the scene and initialize the domino factory.
        run_np = self.models.attach_new_node("domino_run_temp")
        domino_factory = DominoMaker(run_np, world, make_geom=False)

        # First try a shortcut: distribute all dominoes linearly
        # (wrt the parameter).
        thickness, width, height = extents
        length = spl.arclength(tck)
        nb_dom = int(length / (3 * thickness))
        u = np.linspace(0, 1, nb_dom)
        pos = spl.splev3d(u, tck, .5*height)
        head = spl.splang(u, tck)
        # Create the domino run
        for i, (p, h) in enumerate(zip(pos, head)):
            domino_factory.add_domino(
                    Vec3(*p), h, extents, mass=1, prefix="d_{}".format(i))

        if not skip_uniform and self.test_valid_run(run_np, angvel_init):
            # Return coords if valid
            remove_all_bullet_children(run_np.node(), world)
            run_np.remove_node()
            return pos, head
        else:
            # Otherwise remove all dominoes except the first one.
            rmw = world.remove
            rmc = run_np.node().remove_child
            for ni in run_np.node().get_children()[1:]:
                rmw(ni)
                rmc(ni)

        # Now try the regular method.
        # Define constants.
        max_nb_dom = int(length / thickness)
        timestep = 1. / 60.
        max_time = 2.
        base = box(-thickness*.5, -width*.5, thickness*.5,  width*.5)
        # Initialize variables.
        u = [0.]
        doms = []
        last_ustep = 0.
        i = 1
        timeout = False
        # Save the first domino.
        doms.append(run_np.get_child(0))
        doms[0].node().set_angular_velocity(angvel_init)
        # Prepare the cache.
        cache_back = self._physics_cache.copy()
        _add_to_cache(doms[0])

        while 1. - u[-1] > last_ustep and len(u) < max_nb_dom and not timeout:
            # Compute the bounds.
            dx, dy = splev(u[-1], tck, 1)
            ds = np.sqrt(dx*dx + dy*dy)
            lb = u[-1] + thickness / ds  # first-order approximation
            ub = u[-1] + height / ds  # idem
            # Initialize the new domino.
            doms.append(domino_factory.add_domino(
                    0, 0, extents, mass=1, prefix="d_{}".format(i)))
            new_np = doms[-1]
            new = new_np.node()
            _add_to_cache(new_np)
            # Update the last one.
            last_np = doms[-2]
            last = last_np.node()
            last_base = translate(
                    rotate(base, last_np.get_h()),
                    last_np.get_x(), last_np.get_y())

            def objective(ui):
                # Reset simulation.
                reset_physics()
                # Put the new domino at its new position.
                x, y, z = spl.splev3d(ui, tck, .5*height)[0]
                h = spl.splang(ui, tck)
                new_np.set_pos_hpr(x, y, z, h, 0, 0)
                # Make sure the new and last dominoes are not intersecting.
                new_base = translate(rotate(base, h), x, y)
                c = last_base.intersection(new_base).area
                if c > 0:
                    return c
                # Run the simulation until the impact, or until time is up.
                t = 0.
                test = world.contact_test_pair(last, new)
                while test.get_num_contacts() == 0 and t < max_time * i:
                    t += timestep
                    world.do_physics(timestep)
                    test = world.contact_test_pair(last, new)
                nonlocal timeout
                if t >= max_time:
                    timeout = True
                    return 0.
#                    return (ui - u[-1])
                timeout = False
                world.do_physics(timestep)
                angvel = new.get_angular_velocity()
                angvel = angvel.dot(
                        new_np.get_net_transform().get_mat().get_row3(1))
                return -angvel
            # Perform the optimization.
#            res = opt.brute(objective, ranges=[(lb, ub)], Ns=5)
#            ui = res
            res = opt.minimize_scalar(
                    objective, bounds=(lb, ub), method='bounded')
            ui = res.x
            if timeout:
                continue
            # Update the variables.
            u.append(ui)
            last_ustep = ui - u[-2]
            i += 1
            print("Placing domino no. {} at u = {}".format(len(u), ui))
            # Update the state of the new domino in the cache.
            x, y, z = spl.splev3d(ui, tck, .5*height)[0]
            h = spl.splang(ui, tck)
            new_np.set_pos_hpr(x, y, z, h, 0, 0)
            new.set_linear_velocity(0)
            new.set_angular_velocity(0)
            _add_to_cache(new_np)
        # Clean up.
        print("Cleaning up.")
        self._physics_cache = cache_back
        remove_all_bullet_children(run_np.node(), world)
        run_np.remove_node()

        print("Exiting.")
        u = np.asarray(u)
        pos = spl.splev3d(u, tck, .5*height)
        head = spl.splang(u, tck)
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
#  #            u = [0.] + list(u) + [1.]
#  #            avg_dist = sum(
#  #                    ri.centroid.distance(rj.centroid)
#  #                    for ri, rj in zip(rects[:-1], rects[1:])
#  #                    ) / len(rects)
#            return sum(
#                    overlap_energy(ri, rj)
#  #                    + abs(avg_dist - ri.centroid.distance(rj.centroid))
#  #                    / avg_dist
#  #                    + lennard_jones(u[i], u[i+1])
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

    def make_domino_run(self, extents=Vec3(.05, .2, .4)):
        # Get the smooth domino path.
        def project(v):
            p = Point3()
            self.mouse_to_ground((v[0], v[2]), p)
            return p
        tck = spl.get_smooth_path(self.pencil, s=.1, prep=project)

        # Determine the domino distribution.
        angvel_init = Vec3(0., 5., 0.)
        angvel_init = Mat3.rotate_mat(spl.splang(0, tck)).xform(
                angvel_init)
        try:
            pos, head = self.place_dominoes(tck, extents, angvel_init)
        except InvalidPathError:
            print("Invalid path. Aborting.")
        else:
            # Instantiate dominoes.
            print("Instantiating dominoes")
            domino_run_np = self.models.attach_new_node("domino_run")
            domino_factory = DominoMaker(domino_run_np, self.world)
            for i, (p, h) in enumerate(zip(pos, head)):
                domino_factory.add_domino(
                        Vec3(*p), h, extents, mass=1,
                        prefix="domino_{}".format(i))

            first_domino = domino_run_np.get_child(0)
            first_domino.node().set_angular_velocity(angvel_init)
            # Add dominoes to the cache.
            self._create_cache()
        finally:
            # Remove the drawing and show the smoothed curve.
            self.clear_drawing()
            u = np.linspace(0., 1., 100)
            path = spl.show_spline2d(self.render, tck, u, "smoothed path",
                                     color=(1, 0, 0, 1))
            ranges = self.get_valid_path_ranges(tck, extents)
            for i, uj in enumerate(u):
                if any(a <= uj <= b for a, b in ranges):
                    path.set_vertex_color(i, (1, 1, 0, 1))


def main():
    load_prc_file_data("", "win-size 1600 900")
    load_prc_file_data("", "window-title RGM Designer")

    app = DominoPathDrawer()
    app.run()


main()
