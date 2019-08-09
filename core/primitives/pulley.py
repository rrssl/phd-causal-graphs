from panda3d.core import NodePath, Point3, Vec3

from .base import BulletRootNodePath, PrimitiveBase
from .cylinder import Cylinder
from .pivot import Pivot


class Pulley(PrimitiveBase):
    """Create a pulley.

    Parameters
    ----------
    name : string
      Name of the lever.
    extents : float sequence
      Extents of the pulley (same as Cylinder): radius, height.
    pivot_pos : (3,) float sequence
      Relative position of the pivot wrt the primitive.
    pivot_hpr : (3,) float sequence
      Relative orientation of the pivot wrt the primitive.
    pivot_extents : (2,) float sequence
      Parameters of the visual cylinder (if geom is True): radius, height.
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

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
        cyl = Cylinder(name=self.name, extents=self.extents, **self.bt_props)
        cyl_path = cyl.create(geom, phys, path, world, velo)
        pivot = Pivot(
            name=self.name+"_pivot", pivot_pos=self.pivot_pos,
            pivot_hpr=self.pivot_hpr, pivot_extents=self.pivot_extents,
        )
        pivot.create(geom, phys, path, world, [cyl_path])
        return path
