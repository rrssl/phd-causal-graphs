import panda3d.bullet as bt
import solid as sl
from panda3d.core import GeomNode, NodePath, TransformState, Point3, Vec3

from .base import PrimitiveBase
from ..meshio import solid2panda


class Track(PrimitiveBase):
    """Create straight track (e.g. for a ball run).

    The track is square because it makes collision shapes easier.
    The center is the center of the (length, width, height) bounding box.

    Parameters
    ----------
    name : string
      Name of the primitive.
    extents : (4,) float sequence
      Extents of the track: (length, width, height, thickness). The first 3
      are external.

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
            bottom = bt.BulletBoxShape(Vec3(l/2, w/2 - t, t/2))
            body.add_shape(bottom,
                           TransformState.make_pos(Point3(0, 0, (t-h)/2)))
            side = bt.BulletBoxShape(Vec3(l/2, t/2, h/2))
            body.add_shape(side,
                           TransformState.make_pos(Point3(0, (t-w)/2, 0)))
            body.add_shape(side,
                           TransformState.make_pos(Point3(0, (w-t)/2, 0)))
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
        box = sl.cube((l, w, h), center=True)
        groove = sl.cube((l, w - 2*t, h), center=True)
        script = box - sl.translate([0, 0, t])(groove)
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
