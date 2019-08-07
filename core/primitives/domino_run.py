import numpy as np
import panda3d.bullet as bt
from panda3d.core import GeomNode, NodePath, Point3, Vec3

from .base import BulletRootNodePath, PrimitiveBase
from .box import Box
from .cylinder import Cylinder
from ..dominoes import tilt_domino_forward


class DominoRun(PrimitiveBase):
    """Create a domino run.

    Parameters
    ----------
    name : string
      Name of the box.
    extents : float sequence
      Extents of each domino.
    coords : (n,3) ndarray
      (x,y,heading) of each domino.
    tilt_angle : float, optional
      Angle by which the first domino should be tilted. Defaults to 0.

    """

    def __init__(self, name, extents, coords, tilt_angle=0, **bt_props):
        super().__init__(name=name, **bt_props)
        self.extents = extents
        self.coords = np.asarray(coords)
        self.tilt_angle = tilt_angle

    def create(self, geom, phys, parent=None, world=None):
        # Physics
        bodies = []
        if phys:
            shape = bt.BulletBoxShape(Vec3(*self.extents) / 2)
            path = BulletRootNodePath(self.name)
        else:
            path = NodePath(self.name)
        # Geometry
        if geom is not None:
            geom_path = NodePath(
                    Box.make_geom(self.name+"_geom", self.extents))
        #     # Path
        #     path_coords = np.c_[self.coords[:, :2],
        #                         np.zeros(len(self.coords))]
        #     if geom == 'LD':
        #         show_polyline3d(path, path_coords, self.name + "_path")
        #     elif geom == 'HD':
        #         path.attach_new_node(
        #             self.make_path_geom(self.name + "_path", path_coords,
        #                                 n_seg=2**3)
        #         )
        for i, (x, y, head) in enumerate(self.coords):
            name = self.name + "_dom_{}_solid".format(i)
            # Physics
            if phys:
                body = bt.BulletRigidBodyNode(name)
                bodies.append(body)
                body.add_shape(shape)
                self._set_properties(body)
            # Scene graph + local coords
            dom_path = NodePath(body) if phys else NodePath(name)
            dom_path.reparent_to(path)
            dom_path.set_pos(Point3(x, y, self.extents[2] / 2))
            dom_path.set_h(head)
            if i == 0 and self.tilt_angle:
                tilt_domino_forward(dom_path, self.extents, self.tilt_angle)
            # Geometry
            if geom is not None:
                geom_path.instance_to(dom_path)
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_path_geom(name, vertices, thickness=.001, n_seg=2**2):
        geom_node = GeomNode(name)
        vertices = [Vec3(*v) for v in vertices]
        for i, (a, b) in enumerate(zip(vertices[:-1], vertices[1:])):
            name = name + "_seg_" + str(i)
            length = (b - a).length()
            geom = Cylinder.make_geom(name, (thickness, length), False, n_seg)
            path = NodePath(geom)
            path.set_pos(a)
            path.look_at(b)
            path.set_hpr(path, Vec3(90, 0, 90))
            path.flatten_light()
            geom_node.add_geoms_from(path.node())
        return geom_node
