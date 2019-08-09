import math

import panda3d.bullet as bt
import solid as sl
from panda3d.core import GeomNode, NodePath, Point3, TransformState, Vec3

from .base import PrimitiveBase
from ..meshio import solid2panda


class Goblet(PrimitiveBase):
    """Create a goblet.

    Parameters
    ----------
    name : string
      Name.
    extents : (4,) float sequence
      Extents of the goblet / truncated cone (h, r1, r2, eps), as defined in
      solidpython (r1 = radius at the bottom of the cone).

    """

    def __init__(self, name, extents, **bt_props):
        super().__init__(name=name, **bt_props)
        self.extents = extents

    def create(self, geom, phys, parent=None, world=None, velo=None):
        name = self.name + "_solid"
        # Physics
        if phys:
            h, r1, r2, eps = self.extents
            alpha = math.atan2(r1 - r2, h)
            length = math.sqrt((r1 - r2) ** 2 + h ** 2)
            n_seg = 2**5
            body = bt.BulletRigidBodyNode(name)
            self._set_properties(body)
            if velo is not None:
                body.set_linear_velocity(Vec3(*velo[:3]))
                body.set_angular_velocity(Vec3(*velo[3:]))
            # Add bottom
            bottom = bt.BulletCylinderShape(r2, eps)
            bottom.set_margin(eps)
            body.add_shape(bottom,
                           TransformState.make_pos(Point3(0, 0, eps / 2)))
            # Add sides
            side = bt.BulletBoxShape(
                Vec3(eps, 2 * math.pi * r1 / n_seg, length) / 2)
            side.set_margin(eps)
            cz = eps + h/2 - math.cos(alpha) * eps / 2
            cr = (r1 + r2) / 2 + math.sin(alpha) * eps / 2
            for i in range(n_seg):
                ai = (i + .5) * 2 * math.pi / n_seg  # .5 to match the geometry
                pos = Point3(cr * math.cos(ai), cr * math.sin(ai), cz)
                hpr = Vec3(math.degrees(ai), 0, math.degrees(alpha))
                body.add_shape(side, TransformState.make_pos_hpr(pos, hpr))
            bodies = [body]
            path = NodePath(body)
        else:
            bodies = []
            path = NodePath(name)
        # Geometry
        if geom is not None:
            n_seg = 2**5 if geom == 'HD' else 2**4
            path.attach_new_node(
                self.make_geom(self.name+"_geom", self.extents, n_seg)
            )
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_geom(name, extents, n_seg=2**4):
        h, r1, r2, eps = extents
        cos_alpha_inv = math.sqrt(1 + ((r1 - r2) / h)**2)
        h_ext = h + eps
        r1_ext = r1 + eps * cos_alpha_inv
        r2_ext = r1_ext - (r1 - r2) * h_ext / h
        script = (sl.cylinder(r1=r1_ext, r2=r2_ext, h=h_ext, segments=n_seg)
                  - sl.cylinder(r1=r1, r2=r2, h=h, segments=n_seg))
        script = sl.translate([0, 0, h + eps])(sl.rotate([180, 0, 0])(script))
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
