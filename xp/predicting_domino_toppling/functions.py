"""
Functions used to train and evaluate the classifier.

"""
import os
import sys

import numpy as np
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.core import NodePath
from panda3d.core import Vec3

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl
import xp.simulate as simu


def make_box(dims, pos, rot):
    box = BulletRigidBodyNode("box")
    box.add_shape(BulletBoxShape(Vec3(*dims)*.5))
    box.set_static(False)  # otherwise collisions are ignored
    path = NodePath(box)
    path.set_pos(Vec3(*pos))
    path.set_hpr(Vec3(*rot))
    return path


def has_contact(a: NodePath, b: NodePath):
    world = BulletWorld()
    an = a.node()
    bn = b.node()
    world.attach(an)
    world.attach(bn)
    test = world.contact_test_pair(an, bn)
    return test.get_num_contacts() > 0


def run_domino_toppling_xp(params, visual=False):
    """
    Run the domino-pair toppling simulation. If not visual, returns True if
    the second domino topples.

    Parameters
    ----------
    params : sequence
        Parameter vector (x, y, angle), i.e. D2's coordinates relative to D1.
    visual : boolean
        Run the experiment in 'visual' mode, that is, actually display the
        scene in a window. In that case, 'timestep' and 'maxtime' are ignored.
    """
    x, y, a = params
    global_coords = [[0, 0, 0], [x, y, a]]
    doms_np, world = simu.setup_dominoes(global_coords, _make_geom=visual)

    if visual:
        simu.run_simu(doms_np, world, _visual=True)
        return True
    else:
        d1, d2 = doms_np.get_children()
        test = world.contact_test_pair(d1.node(), d2.node())
        if test.get_num_contacts() > 0:
            return False

        times = simu.run_simu(doms_np, world)
        if np.isfinite(times).all():
            return True


def get_rel_coords(u, spline):
    """Get the relative configuration of each domino wrt the previous."""
    # Get local Cartesian coordinates
    # Change origin
    xi, yi = spl.splev(u, spline)
    xi = np.diff(xi)
    yi = np.diff(yi)
    # Rotate by -a_i-1
    ai = spl.splang(u, spline, degrees=False)
    ci_ = np.cos(ai[:-1])
    si_ = np.sin(ai[:-1])
    xi_r = xi*ci_ + yi*si_
    yi_r = -xi*si_ + yi*ci_
    # Get relative angles
    ai = np.degrees(np.diff(ai))
    ai = (ai + 180) % 360 - 180  # Convert from [0, 360) to [-180, 180)
    # Symmetrize
    ai = np.copysign(ai, yi_r)
    yi_r = np.abs(yi_r)

    return np.column_stack((xi_r, yi_r, ai))


def test_contact():
    dims = (.03, .1, .3)
    b1 = make_box(dims, (0, 0, dims[2]*.5), Vec3(0))
    b2 = make_box(dims, (.1, 0, dims[2]*.5), Vec3(0))
    simu.tilt_box_forward(b1, 45)
    assert(has_contact(b1, b2))
    #  from viewers import PhysicsViewer
    #  app = PhysicsViewer()
    #  app.world.attach(b1.node())
    #  app.world.attach(b2.node())
    #  app.run()


def test_domino_toppling_xp():
    assert run_domino_toppling_xp((.02, .01, 15), 0)


if __name__ == "__main__":
    test_contact()
    test_domino_toppling_xp()
