import panda3d.bullet as bt
from panda3d.core import NodePath

from .base import PrimitiveBase


class Empty(PrimitiveBase):
    """Create an empty primitive (useful for constraints & reparametrization).

     Parameters
     ----------
     name : string
       Name of the primitive.

    """
    def __init__(self, name, **bt_props):
        super().__init__(name, **bt_props)

    def create(self, geom, phys, parent=None, world=None):
        name = self.name + "_solid"
        if phys:
            body = bt.BulletRigidBodyNode(name)
            self._set_properties(body)
            bodies = [body]
            path = NodePath(body)
        else:
            bodies = []
            path = NodePath(name)
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_geom(name):
        pass
