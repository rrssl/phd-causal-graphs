import panda3d.bullet as bt
import solid as sl
from panda3d.core import GeomNode, NodePath, Point3, TransformState, Vec3

from .base import PrimitiveBase
from ..meshio import solid2panda


class Cylinder(PrimitiveBase):
    """Create a cylinder.

    Parameters
    ----------
    name : string
      Name of the cylinder.
    extents : float sequence
      Extents of the cylinder: radius, height.
    center : bool
      Whether the cylinder should be centered. Defaults to True.

    """

    def __init__(self, name, extents, center=True,
                 **bt_props):
        super().__init__(name=name, **bt_props)
        self.extents = extents
        self.center = center

    def create(self, geom, phys, parent=None, world=None, velo=None):
        name = self.name + "_solid"
        # Physics
        if phys:
            body = bt.BulletRigidBodyNode(name)
            self._set_properties(body)
            if velo is not None:
                body.set_linear_velocity(Vec3(*velo[:3]))
                body.set_angular_velocity(Vec3(*velo[3:]))
            r, h = self.extents
            shape = bt.BulletCylinderShape(r, h)
            if self.center:
                body.add_shape(shape)
            else:
                body.add_shape(shape,
                               TransformState.make_pos(Point3(0, 0, h/2)))
            bodies = [body]
            path = NodePath(body)
        else:
            bodies = []
            path = NodePath(name)
        # Geometry
        if geom is not None:
            n_seg = 2**5 if geom == 'HD' else 2**4
            path.attach_new_node(
                self.make_geom(
                    self.name + "_geom", self.extents, self.center, n_seg
                )
            )
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_geom(name, extents, center=True, n_seg=2**4):
        r, h = extents
        script = sl.cylinder(r=r, h=h, center=center, segments=n_seg)
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
