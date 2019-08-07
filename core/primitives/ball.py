import panda3d.bullet as bt
import solid as sl
from panda3d.core import GeomNode, NodePath

from .base import PrimitiveBase
from ..meshio import solid2panda


class Ball(PrimitiveBase):
    """Create a ball.

    Parameters
    ----------
    name : string
      Name of the ball.
    radius : float
      Radius of the ball.

    """

    def __init__(self, name, radius, **bt_props):
        super().__init__(name=name, **bt_props)
        self.radius = radius

    def create(self, geom, phys, parent=None, world=None):
        name = self.name + "_solid"
        # Physics
        if phys:
            body = bt.BulletRigidBodyNode(name)
            self._set_properties(body)
            shape = bt.BulletSphereShape(self.radius)
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
                self.make_geom(
                    self.name + "_geom", self.radius, n_seg
                )
            )
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_geom(name, radius, n_seg=2**4):
        script = sl.sphere(radius, segments=n_seg)
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
