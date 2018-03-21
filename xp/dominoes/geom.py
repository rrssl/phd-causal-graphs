"""
Useful geometric operations for dominoes.

"""
import numpy as np
from panda3d.bullet import BulletWorld
from panda3d.core import NodePath, Point3, TransformState, Vec3


def rotate_around(pos: Point3, hpr: Vec3, initial: NodePath):
    """Rotate in place the NodePath around a 3D point.

    Parameters
    ----------
    pos : Point3
      Center of rotation, relative to the NodePath.
    hpr : Vec3
      HPR components of the rotation, relative to the NodePath.
    initial : NodePath
      The NodePath to rotate.

    """
    xform = initial.get_transform()
    xform = xform.compose(
        TransformState.make_pos(pos)
        .compose(TransformState.make_hpr(hpr))
        .compose(TransformState.make_pos(-pos))
    )
    initial.set_transform(xform)


def tilt_box_forward(box: NodePath, angle):
    """Rotate a node with a BulletBoxShape around its locally Y-aligned
    bottom front edge.

    Parameters
    ----------
    box : NodePath
      NodePath to the box.
    angle : float
      Rotation angle around the Y-aligned front edge.

    """
    # Bullet uses a small collision margin for colligion shapes, equal to
    # min(half_dims)/10. This margin is *substracted* from the half extents
    # during the creation of a btBoxShape, so the "true" originally intended
    # size is get_half_extents_with_margin(), not *_without_margin().
    extents = box.node().get_shape(0).get_half_extents_with_margin()
    ctr = Point3(extents[0], 0, -extents[2])
    rotate_around(ctr, Vec3(0, 0, angle), box)


def add_wave(x, y, amp, freq):
    """Add a wave to the original signal. x is supposed strictly increasing.

    y'(x) = y(x) + amp/omega * sin(omega*x)   with omega = freq * pi / x[-1]

    If y is strictly increasing, a differentiable bijective mapping is
    obtained with
      1. |amp| < 1  (ensures that dy'/dx is > 0 for all x)
      2. freq in Z  (ensures that y'[-1] == y[-1])

    """
    if amp * freq == 0:
        return y
    omega = freq * np.pi / x[-1]
    return y + (amp / omega)*np.sin(omega * x)


# TODO. Move outside of dominoes
def has_contact(a: NodePath, b: NodePath, world: BulletWorld=None):
    """Check whether two BulletRigidBodyNodes are in contact.

    Parameters
    ----------
    a : NodePath
      Path to the first node.
    b : NodePath
      Path to the second node.
    world : BulletWorld, optional
      A world where the two nodes are already attached.

    """
    if world is None:
        world = BulletWorld()
        world.attach(a.node())
        world.attach(b.node())
    test = world.contact_test_pair(a.node(), b.node())
    return test.get_num_contacts() > 0
