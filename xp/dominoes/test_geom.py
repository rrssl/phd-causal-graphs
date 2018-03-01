import math

import numpy as np
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import NodePath, Point3, Vec3

from geom import has_contact, rotate_around, tilt_box_forward


def make_dummy_collision_box(extents, pos, rot):
    box = BulletRigidBodyNode("box")
    box.add_shape(BulletBoxShape(Vec3(*extents)*.5))
    box.set_static(False)  # otherwise collisions are ignored
    path = NodePath(box)
    path.set_pos(Point3(*pos))
    path.set_hpr(Vec3(*rot))
    return path


def test_contact_between_two_identical_boxes_first_tilted():
    dims = (.03, .1, .3)
    b1 = make_dummy_collision_box(dims, Point3(0., 0, dims[2]*.5), Vec3(0))
    b2 = make_dummy_collision_box(dims, Point3(.1, 0, dims[2]*.5), Vec3(0))
    tilt_box_forward(b1, 45)
    assert(has_contact(b1, b2))


def test_rotate_cube_by_45_degrees_around_front_edge():
    # Let C be the center of a cube of side 1, at position (0, 0, .5),
    # and rotated by 15 degrees around the Z axis ("heading").
    c = NodePath("")
    c_pos = Point3(0, 0, .5)
    c_hpr = Vec3(30, 0, 0)
    c.set_pos_hpr(c_pos, c_hpr)
    # Let P be a point in the middle of the bottom edge "in front" of the cube.
    p = Point3(.5, 0, -.5)
    # We are going to rotate this cube by 45 degrees around its Y axis,
    # with P as the center of rotation.
    hpr = Vec3(0, 0, 45)
    rotate_around(p, hpr, c)
    # Note that the formula below is only valid when r=45. However, it
    # should work for any c_h.
    expected_pos = Point3(
        .5*math.cos(math.radians(c_hpr[0])),
        .5*math.sin(math.radians(c_hpr[0])),
        math.cos(math.radians(hpr[2]))
    )
    assert np.allclose(c.get_pos(), expected_pos), (
        "Got position {}".format(np.asarray(c.get_pos()))
        + " instead of {}".format(np.asarray(expected_pos))
    )


if __name__ == "__main__":
    test_contact_between_two_identical_boxes_first_tilted()
    test_rotate_cube_by_45_degrees_around_front_edge()
