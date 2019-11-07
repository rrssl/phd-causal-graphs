# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3 (jupyter)
#     language: python
#     name: python3
# ---

# Panda3D's basic geometric transformations are pretty intuitive, but composing them can be somewhat complicated. Here are some examples.

# # Rotating an object around a point

# +
from panda3d.core import NodePath, Point3, TransformState, Vec3

# Let C be the center of a cube of side 1, at position (0, 0, .5),
# and rotated by 15 degrees around the Z axis ("heading"). 
c = NodePath("")
c_pos = Point3(0, 0, .5)
c_hpr = Vec3(15, 0, 0)
c.set_pos_hpr(c_pos, c_hpr)
print("Initial world transform of center: {}".format(c.get_net_transform()))
# Let P be a point in the middle of the bottom edge "in front" of the cube.
p = Point3(.5, 0, -.5)
# We are going to rotate this cube by 45 degrees around its Y axis,
# with P as the center of rotation. 
hpr = Vec3(0, 0, 45)


# -

# ## First version
#
# The first version uses the high-level operations of NodePath. There are probably many ways to do it. This version inserts a pivot as the parent of the original node. The pivot's position is computed from the node's original position + the pivot's local position. The pivot's orientation is alogned to the node's original orientation, and then combines it with the additional rotation. The node's position is now local to the pivot. The node's orientation is the identity (because the pivot received the node's original orientation).

# +
def rotate_around(pos: Point3, hpr: Vec3, initial: NodePath):
    """Rotate in place the NodePath around a 3D point.

    Parameters
    ----------
    pos : Point3
        Center of rotation, relative to the NodePath.
    hpr : Vec3
        HPR components of the rotation, relative to the NodePath's frame.
    initial : NodePath
        The NodePath to rotate.

    """
    pivot = NodePath("pivot")
    # Wrt 'initial', the pivot goes to 'pos', while its frame gets aligned.
    # This is equivalent to
    # pivot.set_pos(initial, pos); pivot.set_hpr(initial.get_hpr())
    pivot.set_pos_hpr(initial, pos, 0)
    print("World transform of the pivot: {}".format(pivot.get_net_transform()))
    # Attach 'initial' to 'pivot' while keeping exactly the same world coordinates.
    initial.wrt_reparent_to(pivot)
    # Rotate pivot by 'hpr'. Note how we add it to the existing rotation.
    pivot.set_hpr(pivot, hpr)

rotate_around(p, hpr, c)
# Note how, under these conditions, C is right above P (cx=px, cy=py)
# and cy=sqrt(2)/2, while the overall rotation is correct (within accuracy).
print("Final world transform of center: {}".format(c.get_net_transform()))
# -

# ## Second version
#
# The second version, more efficient, uses the lower level interface of TransformState. Note how the rotation is composed *to the right* of the original transform, indicating a *local transform*. The `compose` function is equivalent to a matrix product (same order of the operands).

xform = TransformState.make_pos_hpr(c_pos, c_hpr)
xform = xform.compose(
    TransformState.make_pos(p).compose(
    TransformState.make_hpr(hpr)).compose(
    TransformState.make_pos(-p))
)
print("Transform with TransformState operations: {}".format(xform))

# Let us now check the numerical results.

# +
import math

print(.5*math.cos(math.radians(15)), .5*math.sin(math.radians(15)), .5*math.sqrt(2))
