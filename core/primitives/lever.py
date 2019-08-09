from panda3d.core import NodePath, Point3, Vec3

from .base import BulletRootNodePath, PrimitiveBase
from .box import Box
from .pivot import Pivot


class Lever(PrimitiveBase):
    """Create a lever.

    Parameters
    ----------
    name : string
      Name of the lever.
    extents : float sequence
      Extents of the lever (same as Box).
    pivot_pos : (3,) float sequence
      Relative position of the pivot wrt the primitive.
    pivot_hpr : (3,) float sequence
      Relative orientation of the pivot wrt the primitive.
    pivot_extents : (2,) float sequence
      Parameters of the visual cylinder (if geom is True): radius, height.

    """

    def __init__(self, name, extents, pivot_pos, pivot_hpr, pivot_extents=None,
                 **bt_props):
        super().__init__(name)
        self.extents = extents
        self.pivot_pos = Point3(*pivot_pos)
        self.pivot_hpr = Vec3(*pivot_hpr)
        self.pivot_extents = pivot_extents
        self.bt_props = bt_props

    def create(self, geom, phys, parent=None, world=None, velo=None):
        # Scene graph
        path = BulletRootNodePath(self.name) if phys else NodePath(self.name)
        self._attach(path, parent)
        # Physics
        box = Box(name=self.name, extents=self.extents, **self.bt_props)
        box_path = box.create(geom, phys, path, world, velo)
        pivot = Pivot(
            name=self.name + "_pivot", pivot_pos=self.pivot_pos,
            pivot_hpr=self.pivot_hpr, pivot_extents=self.pivot_extents,
        )
        pivot.create(geom, phys, path, world, [box_path])
        return path
