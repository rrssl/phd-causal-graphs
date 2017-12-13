"""
Basic geometric operations on dominoes.

"""
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.core import NodePath
from panda3d.core import Point3, Vec3


def rotate_around(path: NodePath, pivot: Point3, hpr: Vec3):
    """Rotate in place the NodePath around a 3D point.

    Parameters
    ----------
    path : NodePath
        The NodePath to rotate.
    pivot : Point3
        Center of rotation, relative to the NodePath.
    hpr : Vec3
        HPR components of the rotation, relative to the NodePath.
    """
    pivot_np = NodePath("pivot")
    pivot_np.set_pos(path, pivot)
    pivot_np.set_hpr(path.get_hpr())  # Put pivot in the same relative frame
    path.set_hpr(pivot_np, hpr)


def tilt_box_forward(box: NodePath, angle):
    # Bullet uses a small collision margin for colligion shapes, equal to
    # min(half_dims)/10. This margin is *substracted* from the half extents
    # during the creation of a btBoxShape, so the "true" originally intended
    # size is get_half_extents_with_margin(), not *_without_margin().
    extents = box.node().get_shape(0).get_half_extents_with_margin()
    ctr = Point3(extents[0], 0, -extents[2])
    rotate_around(box, ctr, Vec3(0, 0, angle))


def make_collision_box(extents, pos, rot):
    box = BulletRigidBodyNode("box")
    box.add_shape(BulletBoxShape(Vec3(*extents)*.5))
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


def test_contact():
    dims = (.03, .1, .3)
    b1 = make_collision_box(dims, (0, 0, dims[2]*.5), Vec3(0))
    b2 = make_collision_box(dims, (.1, 0, dims[2]*.5), Vec3(0))
    tilt_box_forward(b1, 45)
    assert(has_contact(b1, b2))


if __name__ == "__main__":
    test_contact()
