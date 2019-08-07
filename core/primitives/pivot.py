import panda3d.bullet as bt
from panda3d.core import Point3, TransformState, Vec3

from .base import PrimitiveBase
from .cylinder import Cylinder


class Pivot(PrimitiveBase):
    """Attach a pivot constraint to a primitive.

    Parameters
    ----------
    name : string
      Name of the primitive.
    pivot_pos : (3,) float sequence
      Relative position of the pivot wrt the primitive.
    pivot_hpr : (3,) float sequence
      Relative orientation of the pivot wrt the primitive.
    pivot_extents : None or (2,) float sequence, optional
      Parameters of the visual cylinder (if geom is not None): radius, height.
      None by default.

    """

    def __init__(self, name, pivot_pos, pivot_hpr, pivot_extents=None):
        super().__init__(name)
        pivot_pos = Point3(*pivot_pos)
        pivot_hpr = Vec3(*pivot_hpr)
        self.pivot_xform = TransformState.make_pos_hpr(pivot_pos, pivot_hpr)
        self.pivot_extents = pivot_extents

    def create(self, geom, phys, parent=None, world=None, components=None):
        pivot = Cylinder(name=self.name, extents=self.pivot_extents)
        path = pivot.create(geom, phys, parent, world)
        if not components:
            return path
        path.set_transform(components[0], self.pivot_xform)
        # Physics
        if phys:
            cs = bt.BulletHingeConstraint(
                path.node(), components[0].node(),
                TransformState.make_identity(), self.pivot_xform
            )
            self._attach(constraints=[cs], world=world)
        # return path
