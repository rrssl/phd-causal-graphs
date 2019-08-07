import panda3d.bullet as bt
import solid as sl
from panda3d.core import GeomNode, NodePath, Point3, TransformState, Vec3

from .base import PrimitiveBase
from ..meshio import solid2panda


class OpenBox(PrimitiveBase):
    """Create a hollow box with the top open.

    Parameters
    ----------
    name : string
      Name of the box.
    extents : float sequence
      Extents of the box.

    """

    def __init__(self, name, extents, **bt_props):
        super().__init__(name=name, **bt_props)
        self.extents = extents

    def create(self, geom, phys, parent=None, world=None):
        name = self.name + "_solid"
        # Physics
        if phys:
            body = bt.BulletRigidBodyNode(name)
            self._set_properties(body)
            l, w, h, t = self.extents
            bottom = bt.BulletBoxShape(Vec3(l, w, t) / 2)
            bottom_xform = TransformState.make_pos(Point3(0, 0, t/2-h/2))
            body.add_shape(bottom, bottom_xform)
            front = bt.BulletBoxShape(Vec3(l, h, t) / 2)
            front_xform = TransformState.make_pos_hpr(Point3(0, t/2-w/2, 0),
                                                      Vec3(0, 90, 0))
            body.add_shape(front, front_xform)
            back_xform = TransformState.make_pos_hpr(Point3(0, -t/2+w/2, 0),
                                                     Vec3(0, -90, 0))
            body.add_shape(front, back_xform)
            side = bt.BulletBoxShape(Vec3(h, w, t) / 2)
            left_xform = TransformState.make_pos_hpr(Point3(-t/2+l/2, 0, 0),
                                                     Vec3(0, 0, -90))
            body.add_shape(side, left_xform)
            right_xform = TransformState.make_pos_hpr(Point3(t/2-l/2, 0, 0),
                                                      Vec3(0, 0, 90))
            body.add_shape(side, right_xform)
            bodies = [body]
            path = NodePath(body)
        else:
            bodies = []
            path = NodePath(name)
        # Geometry
        if geom is not None:
            path.attach_new_node(
                self.make_geom(self.name + "_geom", self.extents))
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_geom(name, extents):
        l, w, h, t = extents
        box_out = sl.cube((l, w, h), center=True)
        box_in = sl.cube((l - 2*t, w - 2*t, h), center=True)
        box = box_out - sl.translate([0, 0, t])(box_in)
        geom = solid2panda(box)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
