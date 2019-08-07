import panda3d.bullet as bt
from panda3d.core import Point3, TransformState, Vec3

from .base import PrimitiveBase


class Fastener(PrimitiveBase):
    """Glue two primitives together.

    Parameters
    ----------
    name : string
      Name of the primitive (useless here).
    comp1_xform : (6,) float sequence
      Relative transform of the constraint wrt the first primitive.
    comp2_xform : (6,) float sequence
      Relative transform of the constraint wrt the second primitive.

    """

    def __init__(self, name, comp1_xform, comp2_xform):
        super().__init__(name)
        self.comp1_xform = TransformState.make_pos_hpr(
            Point3(*comp1_xform[:3]), Vec3(*comp1_xform[3:])
        )
        self.comp2_xform = TransformState.make_pos_hpr(
            Point3(*comp2_xform[:3]), Vec3(*comp2_xform[3:])
        )

    def create(self, geom, phys, parent=None, world=None, components=None):
        if not phys or not components:
            return
        comp1, comp2 = components
        cs = bt.BulletGenericConstraint(
            comp1.node(), comp2.node(), self.comp1_xform, self.comp2_xform, 1
        )
        for i in range(3):
            cs.set_angular_limit(i, 0, 0)
            cs.set_linear_limit(i, 0, 0)
        self._attach(constraints=[cs], world=world)
