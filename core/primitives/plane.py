import numpy as np
import panda3d.bullet as bt
from panda3d.core import GeomNode, NodePath, Vec3

from .base import PrimitiveBase
from ..meshio import trimesh2panda


class Plane(PrimitiveBase):
    """Create a plane.

    Parameters
    ----------
    name : string
      Name of the plane.
    normal : (3,) sequence
      Normal to the plane.
    distance : float
      Distance of the plane along the normal.

    """

    def __init__(self, name, normal=(0, 0, 1), distance=0, **bt_props):
        super().__init__(name=name, **bt_props)
        self.normal = Vec3(*normal)
        self.distance = distance

    def create(self, geom, phys, parent=None, world=None):
        name = self.name + "_solid"
        # Physics
        if phys:
            body = bt.BulletRigidBodyNode(name)
            self._set_properties(body)
            shape = bt.BulletPlaneShape(self.normal, self.distance)
            # NB: Using a box instead of a plane might help stability:
            # shape = bt.BulletBoxShape((1, 1, .1))
            # body.add_shape(shape, TransformState.make_pos(Point3(0, 0, -.1)))
            body.add_shape(shape)
            bodies = [body]
            path = NodePath(body)
        else:
            bodies = []
            path = NodePath(name)
        # Geometry
        if geom is not None:
            path.attach_new_node(self.make_geom(
                self.name + "_geom",
                self.normal,
                self.distance
            ))
        self._attach(path, parent, bodies=bodies, world=world)
        return path

    @staticmethod
    def make_geom(name, normal, distance, scale=100):
        # Compute basis
        normal = np.array(normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        tangent = np.ones(3)
        tangent -= tangent.dot(normal) * normal
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        # Compute vertices
        vertices = np.array([
            tangent,
            bitangent,
            -tangent,
            -bitangent
        ]) * scale + distance * normal
        faces = np.array(
            [0, 1, 3, 1, 2, 3],
            dtype=np.int64
        ).reshape(-1, 3)
        vertex_normals = np.tile(normal, (len(vertices), 1))
        geom = trimesh2panda(vertices, faces, vertex_normals)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node
