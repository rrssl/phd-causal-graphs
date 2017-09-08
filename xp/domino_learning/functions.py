"""
Test if two boxes collide.
"""
from math import atan, pi
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.core import NodePath
from panda3d.core import Point3, Vec3

import os, sys
sys.path.insert(0, os.path.abspath("../.."))
from primitives import Floor, DominoMaker


def make_box(dims, pos, rot):
    box = BulletRigidBodyNode("box")
    box.add_shape(BulletBoxShape(Vec3(*dims)*.5))
    box.set_static(False)  # otherwise collisions are ignored
    path = NodePath(box)
    path.set_pos(Vec3(*pos))
    path.set_hpr(Vec3(*rot))
    return path


def rotate_around(path: NodePath, pivot: Point3, hpr: Vec3):
    """Rotate in place the NodePath around a 3D point.

    Parameters
    ----------
    path : NodePath
        The NodePath to rotate.
    pivot : Point3
        Center of rotation, in the relative to the NodePath.
    hpr : Vec3
        HPR components of the rotation, relative to the NodePath.
    """
    pivot_np = NodePath("pivot")
    pivot_np.set_pos(path, pivot)
    pivot_np.set_hpr(path.get_hpr())  # Put pivot in the same relative frame
    path.set_hpr(pivot_np, hpr)


def tilt_box_forward(box: NodePath, angle):
    ext = box.node().get_shape(0).get_half_extents_with_margin()
    ctr = Point3(ext[0], 0, -ext[2])
    rotate_around(box, ctr, Vec3(0, 0, angle))


def has_contact(a: NodePath, b: NodePath):
    world = BulletWorld()
    an = a.node()
    bn = b.node()
    world.attach(an)
    world.attach(bn)
    test = world.contact_test_pair(an, bn)
    return test.get_num_contacts() > 0


# TODO. Apply improvements discovered in domino_design: activate cache
# flushing, run until dominoes are deactivated. Or better, create a common
# class.
def run_domino_toppling_xp(params, timestep, maxtime, visual=False):
    """
    Run the domino-pair toppling simulation. If not visual, returns True if
    the second domino topples.

    Parameters
    ----------
    params : sequence
        Parameter vector (thickness, width, height, x, y, angle, mass).
        (x, y, angle) are D2's coordinates relative to D1.
    timestep : float
        Simulation timestep.
    maxtime : float
        Maximum simulation time without d2 toppling.
    visual : boolean
        Run the experiment in 'visual' mode, that is, actually display the
        scene in a window. In that case, 'timestep' and 'maxtime' are ignored.
    """
    # World
    world = BulletWorld()
    world.set_gravity(Vec3(0, 0, -9.81))
    # Floor
    floor_path = NodePath("floor")
    floor = Floor(floor_path, world)
    floor.create()
    # Dominoes
    dom_path = NodePath("dominoes")
    dom_fact = DominoMaker(dom_path, world, make_geom=visual)
    t, w, h, x, y, a, m = params
    d1 = dom_fact.add_domino(Vec3(0, 0, h*.5), 0, Vec3(t, w, h), m, "d1")
    d2 = dom_fact.add_domino(Vec3(x, y, h*.5), a, Vec3(t, w, h), m, "d2")
    # Initial state
    toppling_angle = atan(t / h) * 180 / pi + 1
    tilt_box_forward(d1, toppling_angle)
    d1.node().set_transform_dirty()

    if visual:
        from viewers import PhysicsViewer
        app = PhysicsViewer()
        dom_path.reparent_to(app.models)
        app.world = world
        try:
            app.run()
        except SystemExit:
            app.destroy()
        return True
    else:
        t = 0.
        while d2.get_r() < toppling_angle and t < maxtime:
            t += timestep
            world.do_physics(timestep)

        return d2.get_r() >= toppling_angle


def test_contact():
    dims = (.03, .1, .3)
    b1 = make_box(dims, (0, 0, dims[2]*.5), Vec3(0))
    b2 = make_box(dims, (.1, 0, dims[2]*.5), Vec3(0))
    tilt_box_forward(b1, 45)
    assert(has_contact(b1, b2))
    #  import os, sys
    #  sys.path.insert(0, os.path.abspath(".."))
    #  from viewers import PhysicsViewer
    #  app = PhysicsViewer()
    #  app.world.attach(b1.node())
    #  app.world.attach(b2.node())
    #  app.run()


def test_domino_toppling_xp():
    assert run_domino_toppling_xp((.03, .1, .3, .1, 0, 15, .1), 1/60, 1)


if __name__ == "__main__":
    #  test_contact()
    test_domino_toppling_xp()
