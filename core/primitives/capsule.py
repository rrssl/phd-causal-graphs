import panda3d.bullet as bt
import solid as sl
import solid.utils as slu
from panda3d.core import GeomNode, NodePath, Vec3

from .base import PrimitiveBase
from ..meshio import solid2panda


class Capsule(PrimitiveBase):
    """Create a capsule.

    Parameters
    ----------
    name : string
      Name of the capsule.
    extents : float sequence
      Extents of the capsule: radius, height.

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
            r, h = self.extents
            shape = bt.BulletCapsuleShape(r, h)
            body.add_shape(shape)
            bodies = [body]
            path = NodePath(body)
        else:
            bodies = []
            path = NodePath(name)
        # Geometry
        if geom is not None:
            n_seg = 2**5 if geom == 'HD' else 2**4
            path.attach_new_node(
                self.make_geom(self.name + "_geom", self.extents, n_seg)
            )
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_geom(name, extents, n_seg=2**4):
        r, h = extents
        ball = sl.sphere(r=r, segments=n_seg)
        script = (sl.cylinder(r=r, h=h, center=True, segments=n_seg)
                  + slu.up(h / 2)(ball)
                  + slu.down(h / 2)(ball)
                  )
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
