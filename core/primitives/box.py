import panda3d.bullet as bt
import solid as sl
from panda3d.core import GeomNode, NodePath, Vec3

from .base import PrimitiveBase
from ..meshio import solid2panda


class Box(PrimitiveBase):
    """Create a box.

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

    def create(self, geom, phys, parent=None, world=None, velo=None):
        name = self.name + "_solid"
        # Physics
        if phys:
            body = bt.BulletRigidBodyNode(name)
            self._set_properties(body)
            if velo is not None:
                body.set_linear_velocity(Vec3(*velo[:3]))
                body.set_angular_velocity(Vec3(*velo[3:]))
            shape = bt.BulletBoxShape(Vec3(*self.extents) / 2)
            #  shape.set_margin(.0001)
            body.add_shape(shape)
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
        box = sl.cube(tuple(extents), center=True)
        geom = solid2panda(box)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
