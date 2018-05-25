"""
Basic primitives for the RGMs.

"""
import math
from functools import partial

import numpy as np
import panda3d.bullet as bt
import solid as sl
import solid.utils as slu
from panda3d.core import (GeomNode, LineSegs, NodePath, Point3,
                          PythonCallbackObject, TransformState, Vec3)

from core.meshio import solid2panda, trimesh2panda


class World(bt.BulletWorld):
    """The world in which the primitives live."""

    def __init__(self):
        super().__init__()
        # Trick to have several physics callbacks.
        self._callbacks = []
        self.set_tick_callback(
            PythonCallbackObject(self._run_callbacks),
            is_pretick=True
        )

    def _run_callbacks(self, callback_data):
        for cb in self._callbacks:
            cb(callback_data)
        callback_data.upcall()  # just to be safe

    def set_gravity(self, gravity):
        gravity = Vec3(*gravity)
        super().set_gravity(gravity)


class BulletRootNodePath(NodePath):
    """Special NodePath, parent to bt nodes, that propagates transforms."""

    def __init__(self, *args):
        super().__init__(*args)

        xforms = ['set_pos', 'set_hpr', 'set_pos_hpr',
                  'set_x', 'set_y', 'set_z', 'set_h', 'set_p', 'set_r']

        for xform in xforms:
            setattr(self, xform,
                    partial(self.propagate_xform, xform=xform))

    def propagate_xform(self, *args, xform=''):
        getattr(super(), xform)(*args)
        for child in self.get_children():
            if isinstance(child.node(), bt.BulletBodyNode):
                child.node().set_transform_dirty()


class PrimitiveBase:
    """Base class for all primitives.

     Parameters
     ----------
     name : string
       Name of the primitive.
     geom : bool
       Whether to generate a geometry for visualization.
     bt_props : dict
       Dictionary of Bullet properties (mass, restitution, etc.). Basically
       the method set_key is called for the Bullet body, where "key" is each
       key of the dictionary.

    """

    def __init__(self, name, geom=False, **bt_props):
        self.name = name
        self.bt_props = bt_props
        self.geom = geom

        self.path = None
        self.bodies = []
        self.constraints = []
        self.physics_callback = None

    def attach_to(self, path: NodePath, world: World):
        """Attach the object to the scene.

        Parameters
        ----------
        path : NodePath
            Path of the node in the scene tree where where objects are added.
        world : World
            Physical world where the rigid bodies and constraints are added.

        """
        self.path.reparent_to(path)
        for body in self.bodies:
            world.attach(body)
        for cs in self.constraints:
            world.attach_constraint(cs, linked_collision=True)
            cs.set_debug_draw_size(.05)
        if self.physics_callback is not None:
            world._callbacks.append(self.physics_callback)

    def create(self):
        raise NotImplementedError

    def _set_properties(self, bullet_object):
        for key, value in self.bt_props.items():
            getattr(bullet_object, "set_" + key)(value)


class Empty(PrimitiveBase):
    """Create an empty primitive (useful for constraints).

     Parameters
     ----------
     name : string
       Name of the primitive.

    """

    def __init__(self, name):
        super().__init__(name=name)

    def create(self):
        body = bt.BulletRigidBodyNode(self.name)
        self.bodies.append(body)
        self.path = NodePath(body)
        return self.path

    @staticmethod
    def make_geom(name):
        pass


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
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, normal=(0, 0, 1), distance=0, geom=False,
                 **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.normal = normal
        self.distance = distance

    def create(self):
        # Physics
        body = bt.BulletRigidBodyNode(self.name + "_solid")
        self.bodies.append(body)
        self._set_properties(body)
        # TODO. Investigate whether PlaneShape really is the cause
        # of the problem. Maybe it's just a matter of collision margin?
        # shape = bt.BulletBoxShape((1, 1, .1))
        # body.add_shape(shape, TransformState.make_pos(Point3(0, 0, -.1)))
        shape = bt.BulletPlaneShape(Vec3(*self.normal), self.distance)
        body.add_shape(shape)
        # Scene graph
        self.path = NodePath(body)
        # Geometry
        if self.geom:
            self.path.attach_new_node(self.make_geom(
                self.name + "_geom",
                self.normal,
                self.distance
            ))
        return self.path

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
        vertices = np.array(
            [-tangent - bitangent,
             -tangent + bitangent,
             tangent + bitangent,
             tangent - bitangent]
        ) * scale + distance * normal
        faces = np.array(
            [0, 1, 3, 1, 2, 3],
            dtype=np.int64
        ).reshape(-1, 3)
        vertex_normals = np.array(
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            dtype=np.float64
        ).reshape(-1, 3)
        geom = trimesh2panda(vertices, faces, vertex_normals)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node


class Ball(PrimitiveBase):
    """Create a ball.

    Parameters
    ----------
    name : string
      Name of the ball.
    radius : float
      Radius of the ball.
    geom : bool
       Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, radius, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.radius = radius

    def create(self):
        # Physics
        body = bt.BulletRigidBodyNode(self.name + "_solid")
        self.bodies.append(body)
        self._set_properties(body)
        shape = bt.BulletSphereShape(self.radius)
        body.add_shape(shape)
        # Scene graph
        self.path = NodePath(body)
        # Geometry
        if self.geom:
            self.path.attach_new_node(
                self.make_geom(self.name + "_geom", self.radius))
        return self.path

    @staticmethod
    def make_geom(name, radius, n_seg=2 ** 4):
        script = sl.sphere(radius, segments=n_seg)
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node


class Box(PrimitiveBase):
    """Create a box.

    Parameters
    ----------
    name : string
      Name of the box.
    extents : float sequence
      Extents of the box.
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, extents, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents

    def create(self):
        # Physics
        body = bt.BulletRigidBodyNode(self.name + "_solid")
        self.bodies.append(body)
        self._set_properties(body)
        shape = bt.BulletBoxShape(Vec3(*self.extents) / 2)
        #  shape.set_margin(.0001)
        body.add_shape(shape)
        # Scene graph
        self.path = NodePath(body)
        # Geometry
        if self.geom:
            self.path.attach_new_node(
                self.make_geom(self.name + "_geom", self.extents))
        return self.path

    @staticmethod
    def make_geom(name, extents):
        box = sl.cube(tuple(extents), center=True)
        geom = solid2panda(box)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node


class Cylinder(PrimitiveBase):
    """Create a cylinder.

    Parameters
    ----------
    name : string
      Name of the cylinder.
    extents : float sequence
      Extents of the cylinder: radius, height.
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, extents, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents

    def create(self):
        # Physics
        body = bt.BulletRigidBodyNode(self.name)
        self.bodies.append(body)
        self._set_properties(body)
        r, h = self.extents
        shape = bt.BulletCylinderShape(r, h)
        body.add_shape(shape)
        # Scene graph
        self.path = NodePath(body)
        # Geometry
        if self.geom:
            self.path.attach_new_node(
                self.make_geom(self.name + "_geom", self.extents))
        return self.path

    @staticmethod
    def make_geom(name, extents, n_seg=2 ** 4):
        r, h = extents
        script = sl.cylinder(r=r, h=h, center=True, segments=n_seg)
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node


class Capsule(PrimitiveBase):
    """Create a capsule.

    Parameters
    ----------
    name : string
      Name of the capsule.
    extents : float sequence
      Extents of the capsule: radius, height.
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, extents, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents

    def create(self):
        # Physics
        body = bt.BulletRigidBodyNode(self.name)
        self.bodies.append(body)
        self._set_properties(body)
        r, h = self.extents
        shape = bt.BulletCapsuleShape(r, h)
        body.add_shape(shape)
        # Scene graph
        self.path = NodePath(body)
        # Geometry
        if self.geom:
            self.path.attach_new_node(
                self.make_geom(self.name + "_geom", self.extents))
        return self.path

    @staticmethod
    def make_geom(name, extents, n_seg=2 ** 4):
        r, h = extents
        ball = sl.sphere(r=r, segments=n_seg)
        script = (sl.cylinder(r=r, h=h, center=True, segments=n_seg)
                  + sl.translate(slu.up(h / 2))(ball)
                  + sl.translate(slu.down(h / 2))(ball)
                  )
        geom = solid2panda(script)
        geom_node = GeomNode(name)
        geom_node.add_geom(geom)
        return geom_node


class Pivot(PrimitiveBase):
    """Attach a pivot constraint to a primitive.

    Parameters
    ----------
    name : string
      Name of the primitive.
    obj : PrimitiveBase
      Primitive attached to the pivot.
    pivot_pos : (3,) float sequence
      Relative position of the pivot wrt the primitive.
    pivot_hpr : (3,) float sequence
      Relative orientation of the pivot wrt the primitive.
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, obj: PrimitiveBase, pivot_pos, pivot_hpr,
                 geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.obj = obj
        self.pivot_xform = TransformState.make_pos_hpr(pivot_pos, pivot_hpr)

    def create(self):
        # Physics
        pivot = Empty(name=self.name)
        pivot.create().set_transform(self.obj.path, self.pivot_xform)
        self.bodies += pivot.bodies
        cs = bt.BulletHingeConstraint(
            pivot.bodies[0], self.obj.bodies[0],
            TransformState.make_identity(), self.pivot_xform
        )
        self.constraints.append(cs)
        # Scene graph
        self.path = pivot.path
        return self.path


class Lever(PrimitiveBase):
    """Create a lever.

    Parameters
    ----------
    name : string
      Name of the lever.
    extents : float sequence
      Extents of the lever (same as Box).
    pivot_pos_hpr : float sequence
      Relative position and orientation of the pivot (X, Y, Z, H, P, R).
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, extents, pivot_pos_hpr, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents
        self.pivot_pos = Point3(*pivot_pos_hpr[:3])
        self.pivot_hpr = Vec3(*pivot_pos_hpr[3:])

    def create(self):
        # Physics
        box = Box(name=self.name, extents=self.extents, geom=self.geom,
                  **self.bt_props)
        box.create()
        self.bodies += box.bodies
        pivot = Empty(name=self.name + "_pivot")
        pivot.create()
        self.bodies += pivot.bodies
        frame = TransformState.make_pos_hpr(self.pivot_pos, self.pivot_hpr)
        cs = bt.BulletHingeConstraint(
                box.bodies[0], pivot.bodies[0], frame, frame)
        self.constraints.append(cs)
        # Scene graph
        self.path = BulletRootNodePath(pivot.bodies[0])
        box.path.reparent_to(self.path)
        return self.path


class Pulley(PrimitiveBase):
    """Create a pulley.

    Parameters
    ----------
    name : string
      Name of the lever.
    extents : float sequence
      Extents of the pulley (same as Cylinder).
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, extents, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents

    def create(self):
        # Physics
        cyl = Cylinder(name=self.name, extents=self.extents, geom=self.geom,
                       **self.bt_props)
        cyl.create()
        self.bodies += cyl.bodies
        pivot = Empty(name=self.name+"_pivot")
        pivot.create()
        self.bodies += pivot.bodies
        frame = TransformState.make_hpr(Vec3(0, 0, 90))
        cs = bt.BulletHingeConstraint(
                cyl.bodies[0], pivot.bodies[0], frame, frame)
        self.constraints.append(cs)
        # Scene graph
        self.path = BulletRootNodePath(pivot.bodies[0])
        cyl.path.reparent_to(self.path)
        return self.path


class Goblet(PrimitiveBase):
    """Create a goblet.

    Parameters
    ----------
    name : string
      Name.
    extents : float sequence
      Extents of the goblet / truncated cone (h, r1, r2), as defined in
      solidpython (r1 = radius at the bottom of the cone).
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, extents, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents

    def create(self):
        h, r1, r2 = self.extents
        alpha = math.atan2(r1 - r2, h)
        length = math.sqrt((r1 - r2) ** 2 + h ** 2)
        eps = 1e-3
        n_seg = 2 ** 4
        # Physics
        body = bt.BulletRigidBodyNode(self.name + "_solid")
        self.bodies.append(body)
        self._set_properties(body)
        # Add bottom
        bottom = bt.BulletCylinderShape(r2, eps)
        bottom.set_margin(eps)
        body.add_shape(bottom, TransformState.make_pos(Point3(0, 0, eps / 2)))
        # Add sides
        side = bt.BulletBoxShape(
            Vec3(eps, 2 * math.pi * r1 / n_seg, length) / 2)
        cz = eps + h/2 - math.cos(alpha) * eps / 2
        cr = (r1 + r2) / 2 + math.sin(alpha) * eps / 2
        for i in range(n_seg):
            ai = (i + .5) * 2 * math.pi / n_seg  # .5 to match the geometry
            pos = Point3(cr * math.cos(ai), cr * math.sin(ai), cz)
            hpr = Vec3(math.degrees(ai), 0, math.degrees(alpha))
            body.add_shape(side, TransformState.make_pos_hpr(pos, hpr))
        # Scene graph
        self.path = NodePath(body)
        # Geometry
        if self.geom:
            self.path.attach_new_node(
                    self.make_geom(self.name+"_geom", self.extents, n_seg))
        return self.path

    @staticmethod
    def make_geom(name, extents, n_seg=2**4):
        h, r1, r2 = extents
        cos_alpha_inv = math.sqrt(1 + ((r1 - r2) / h)**2)
        eps = 1e-3
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
    geom : bool
      True if a visible geometry should be added to the scene.

    """

    def __init__(self, name, extents, coords, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents
        self.coords = coords

    def create(self):
        # Physics
        shape = bt.BulletBoxShape(Vec3(*self.extents) / 2)
        # Scene graph
        self.path = NodePath(self.name)
        # Geometry
        if self.geom:
            geom_path = NodePath(
                    Box.make_geom(self.name+"_geom", self.extents))

        for i, (x, y, head) in enumerate(self.coords):
            # Physics
            body = bt.BulletRigidBodyNode(self.name + "_body_{}".format(i))
            self.bodies.append(body)
            body.add_shape(shape)
            self._set_properties(body)
            # Scene graph + local coords
            dom_path = self.path.attach_new_node(body)
            dom_path.set_pos(Point3(x, y, self.extents[2] / 2))
            dom_path.set_h(head)
            # Geometry
            if self.geom:
                geom_path.instance_to(dom_path)
        return self.path


class RopePulley(PrimitiveBase):
    """Create a rope-pulley system connecting two primitives.

    Parameters
    ----------
    name : string
      Name of the primitive.
    obj1 : PrimitiveBase
      Primitive connected to the first end of the rope.
    obj2 : PrimitiveBase
      Primitive connected to the second end of the rope.
    obj1_pos : (3,) float sequence
      Relative position of the hook on obj1.
    obj2_pos : (3,) float sequence
      Relative position of the hook on obj2.
    rope_length : float
      Length of the rope.
    pulley_coords : (n,3) float array
      (x, y, z) of each pulley.
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name,
                 obj1: PrimitiveBase, obj2: PrimitiveBase,
                 obj1_pos, obj2_pos,
                 rope_length, pulley_coords,
                 geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.obj1 = obj1
        self.obj2 = obj2
        self.obj1_pos = Point3(*obj1_pos)
        self.obj2_pos = Point3(*obj2_pos)
        self.rope_length = rope_length
        self.pulley_coords = [Point3(*c) for c in pulley_coords]
        # Useful values.
        self.dist_between_pulleys = sum(
            (c2 - c1).length()
            for c2, c1 in zip(self.pulley_coords[1:], self.pulley_coords[:-1])
        )
        self.max_dist = self.rope_length - self.dist_between_pulleys
        assert self.max_dist > 0
        # Hardcoded physical properties.
        self.hook_inertia = 1e-3
        self.max_slider_force = 1e6

    def _attach_objects(self):
        self._attach_pulley(self.obj1, self.obj1_pos, self.pulley_coords[0])
        self._attach_pulley(self.obj2, self.obj2_pos, self.pulley_coords[-1])

    def _attach_pulley(self, target, target_coords, pulley_coords):
        target.bodies[0].set_deactivation_enabled(False)
        target_name = target.path.get_name()
        object_hook_coords = target.path.get_transform(
        ).get_mat().xform_point(target_coords)
        # Each pulley connection is a combination of three constraints:
        # One point-to-point at the pulley, another at the target, and a slider
        # between them.
        # Pulley base (static)
        pulley_base = Empty(name=target_name + "_pulley-base")
        pulley_base.create().set_pos(pulley_coords)
        self.bodies += pulley_base.bodies
        pulley_base.path.reparent_to(self.path)
        # Pulley hook (can rotate on the base)
        pulley_hook = Empty(name=target_name + "_pulley-hook")
        pulley_hook.create().set_pos(pulley_coords)
        pulley_hook.path.look_at(object_hook_coords)  # Y looks at
        pulley_hook.path.set_hpr(pulley_hook.path, Vec3(90, 0, 0))  # now X
        pulley_hook.bodies[0].set_mass(1e-2)  # make it dynamic
        pulley_hook.bodies[0].set_inertia(self.hook_inertia)  # allow rotation
        self.bodies += pulley_hook.bodies
        pulley_hook.path.reparent_to(self.path)
        # Object hook (can rotate on the object)
        object_hook = Empty(name=target_name + "_object-hook")
        object_hook.create().set_pos(object_hook_coords)
        object_hook.path.look_at(pulley_hook.path.get_pos())  # Y looks at
        object_hook.path.set_hpr(object_hook.path, Vec3(-90, 0, 0))  # now -X
        object_hook.bodies[0].set_mass(1e-3)  # make it dynamic
        object_hook.bodies[0].set_inertia(self.hook_inertia)  # allow rotation
        self.bodies += object_hook.bodies
        object_hook.path.reparent_to(self.path)
        # Constraints
        cs1 = bt.BulletSphericalConstraint(
            pulley_base.bodies[0], pulley_hook.bodies[0],
            Point3(0), Point3(0)
        )
        self.constraints.append(cs1)
        cs2 = bt.BulletSliderConstraint(  # along the X-axis by default
            pulley_hook.bodies[0], object_hook.bodies[0],
            TransformState.make_pos(0), TransformState.make_pos(0), True
        )
        cs2.set_lower_linear_limit(0)
        cs2.set_upper_linear_limit(self.max_dist)
        cs2.set_max_linear_motor_force(self.max_slider_force)
        self.constraints.append(cs2)
        cs3 = bt.BulletSphericalConstraint(
            object_hook.bodies[0], target.bodies[0],
            Point3(0), target_coords
        )
        self.constraints.append(cs3)

    def _apply_rope_tension(self, callback_data):
        slider1 = self.constraints[1]
        slider2 = self.constraints[4]
        # If in tension now
        if self._get_loose_rope_length() <= 0:
            # If in tension before
            if self._in_tension:
                mass1 = self.obj1.bodies[0].get_mass()
                mass2 = self.obj2.bodies[0].get_mass()
                hook1 = self.constraints[2]
                hook2 = self.constraints[5]
                imp1 = -hook1.get_applied_impulse()
                imp2 = -hook2.get_applied_impulse()
                step = callback_data.get_timestep()
                # No idea why this formula works.
                delta = 9.81 * step * (imp1 - imp2) / (mass1 + mass2)
                new_dist1 = slider1.get_upper_linear_limit() + delta
                # Clamp value between hard limits.
                if new_dist1 < 0:
                    new_dist1 = 0
                elif new_dist1 > self.max_dist:
                    new_dist1 = self.max_dist
                new_dist2 = slider2.get_upper_linear_limit() - delta
                # Clamp value between hard limits.
                if new_dist2 < 0:
                    new_dist2 = 0
                elif new_dist2 > self.max_dist:
                    new_dist2 = self.max_dist
            else:
                self._in_tension = True
                new_dist1 = slider1.get_linear_pos()
                new_dist2 = slider2.get_linear_pos()
            slider1.set_upper_linear_limit(new_dist1)
            slider2.set_upper_linear_limit(new_dist2)
        # If in tension before but not anymore
        elif self._in_tension:
            self._in_tension = False
            slider1.set_upper_linear_limit(self.max_dist)
            slider2.set_upper_linear_limit(self.max_dist)
        # (If not in tension now or before, don't update anything.)
        if self.geom:
            self._update_visual_rope()

    def _get_loose_rope_length(self):
        # Better use the points than slider.get_linear_pos() because the
        # latter is not initialized yet at the first frame.
        dist1 = (
            self.bodies[2].get_transform().get_pos() - self.pulley_coords[0]
        ).length()
        dist2 = (
            self.bodies[-1].get_transform().get_pos() - self.pulley_coords[-1]
        ).length()
        loose_rope_length = self.rope_length - (
            self.dist_between_pulleys + dist1 + dist2
        )
        return loose_rope_length

    def _get_pulley_acc(self):
        gravity = 9.81
        mass1 = self.obj1.bodies[0].get_mass()
        mass2 = self.obj2.bodies[0].get_mass()
        return gravity * (mass1-mass2) / (mass1+mass2)

    def _get_pulley_hpr(self):
        pulley_line = self.pulley_coords[-1] - self.pulley_coords[0]
        if pulley_line[0]:
            pulley_hpr = Vec3(0, 90, 0)
        else:
            pulley_hpr = Vec3(0, 0, 90)
        return pulley_hpr

    def _update_visual_rope(self):
        P1 = self.bodies[2].get_transform().get_pos()
        P2 = self.pulley_coords[0]
        Pn_1 = self.pulley_coords[-1]
        Pn = self.bodies[-1].get_transform().get_pos()
        loose_rope_length = self._get_loose_rope_length()
        vertices = [P1, P2]
        if loose_rope_length > 0:
            t = np.linspace(0, 1, 21)[1:-1]
            for ti in t:
                p = P2 * (1-ti) + Pn_1 * ti
                p[2] -= loose_rope_length * .5 * math.sin(math.pi * ti)
                vertices.append(p)
        vertices += [Pn_1, Pn]
        vertiter = iter(vertices)
        # Replace old rope with new
        name = "rope"
        old = self.path.find(name)
        if not old.is_empty():
            old.remove_node()
        ls = LineSegs(name)
        ls.set_thickness(5)
        ls.set_color(0)
        ls.move_to(next(vertiter))
        for v in vertiter:
            ls.draw_to(v)
        self.path.attach_new_node(ls.create())

    def create(self):
        # Scene graph
        self.path = NodePath(self.name)
        # Physics
        self._attach_objects()
        self._pulley_acc = self._get_pulley_acc()
        self._in_tension = False
        self.physics_callback = self._apply_rope_tension
        # Geometry
        if self.geom:
            self._update_visual_rope()
            # Pulley geometry
            pulley_hpr = self._get_pulley_hpr()
            for i, coords in enumerate(self.pulley_coords):
                pulley = self.path.attach_new_node(
                    Cylinder.make_geom("pulley"+str(i)+"_geom", (.003, .05))
                )
                pulley.set_pos(coords)
                pulley.set_hpr(pulley_hpr)
        return self.path


class RopePulleyPivot(RopePulley):
    """Create a rope-pulley system connecting two primitives, with a pivot at
    the end.

    Parameters
    ----------
    name : string
      Name of the primitive.
    obj1 : PrimitiveBase
      Primitive connected to the first end of the rope.
    obj2 : PrimitiveBase
      Primitive connected to the second end of the rope.
    obj1_pos : (3,) float sequence
      Relative position of the hook on obj1.
    obj2_pos : (3,) float sequence
      Relative position of the pivot wrt obj2.
    rope_length : float
      Length of the rope.
    pulley_coords : (n,3) float array
      (x, y, z) of each pulley.
    pivot_extents : (2,) float sequence
      Height and radius of the pivot.
    rot_dir : {-1,1}
      Whether tugging the rope results in a [1] direct (counterclockwise) or
      [-1] indirect (clockwise) rotation of the pivot.
    coiled_length : float
      How much length of the rope is initially coiled around the pivot.
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name,
                 obj1: PrimitiveBase, obj2: PrimitiveBase,
                 obj1_pos, obj2_pos,
                 rope_length, pulley_coords,
                 pivot_extents, rot_dir, coiled_length,
                 geom=False, **bt_props):
        super().__init__(name, obj1, obj2, obj1_pos, obj2_pos, rope_length,
                         pulley_coords, geom=geom, **bt_props)
        self.pivot_extents = pivot_extents
        self.rot_dir = rot_dir
        self.init_coiled_length = coiled_length
        # Useful values.
        self.max_dist -= (
            obj2.path.get_transform().get_mat().xform_point(obj2_pos)
            - pulley_coords[-1]
        ).length()
        self.max_angle = math.degrees(
            self.init_coiled_length / self.pivot_extents[1]
        )
        # Hardcoded physical properties.
        self.hinge_force = 1

    def _attach_objects(self):
        self._attach_pulley(self.obj1, self.obj1_pos, self.pulley_coords[0])
        self._attach_pivot(self.obj2, self.obj2_pos)

    def _attach_pivot(self, target, target_coords):
        target.bodies[0].set_deactivation_enabled(False)
        target_name = target.path.get_name()
        pivot_pos = target.path.get_transform(
        ).get_mat().xform_point(target_coords)
        # Empty to support the hinge.
        pivot = Cylinder(name=target_name + "_pivot",
                         extents=self.pivot_extents, geom=self.geom)
        pulley_hpr = self._get_pulley_hpr()
        pivot.create().set_pos_hpr(pivot_pos, pulley_hpr)
        self.bodies += pivot.bodies
        pivot.path.reparent_to(self.path)
        # Constraint
        cs = bt.BulletHingeConstraint(
            pivot.bodies[0], target.bodies[0],
            TransformState.make_pos_hpr(Point3(0), Vec3(0)),
            TransformState.make_pos_hpr(target_coords, pulley_hpr)
        )
        init_hinge_angle = cs.get_hinge_angle()
        cs.set_limit(*sorted(
            [init_hinge_angle,
             init_hinge_angle + self.rot_dir * self.max_angle]
        ))
        self.constraints.append(cs)
        # Some bookkeeping to update rope tension
        self.init_dist2 = self.init_coiled_length + (
            pivot_pos - self.pulley_coords[-1]
        ).length()
        self.init_hinge_angle = cs.get_hinge_angle()

    def _apply_rope_tension(self, callback_data):
        slider = self.constraints[1]
        hinge = self.constraints[3]
        step = callback_data.get_timestep()
        if self._get_loose_rope_length() <= 0:
            if not slider.get_powered_linear_motor():
                # Turn on the motor
                slider.set_powered_linear_motor(True)
            step = callback_data.get_timestep()
            old_velocity = slider.get_target_linear_motor_velocity()
            velocity = old_velocity + self._pulley_acc * step
            slider.set_target_linear_motor_velocity(velocity)
            hinge.enable_angular_motor(
                True, -velocity/self.pivot_extents[1], self.hinge_force * step
            )
        else:
            if slider.get_powered_linear_motor():
                # Turn off the motor
                slider.set_powered_linear_motor(False)
                slider.set_target_linear_motor_velocity(0)
            # We want the hinge to be still when there is no tension.
            hinge.enable_angular_motor(True, 0, self.hinge_force * step)
        if self.geom:
            self._update_visual_rope()

    def _get_loose_rope_length(self):
        # Better use the points than slider.get_linear_pos() because the
        # latter is not initialized yet at the first frame.
        dist1 = (
            self.bodies[2].get_transform().get_pos() - self.pulley_coords[0]
        ).length()
        dist2 = self.init_dist2 - (
            self.rot_dir * self.pivot_extents[1]
            * math.radians(self.constraints[3].get_hinge_angle()
                           - self.init_hinge_angle)
        )
        loose_rope_length = self.rope_length - (
            self.dist_between_pulleys + dist1 + dist2
        )
        return loose_rope_length


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
    geom : bool
      Whether to generate a geometry for visualization.
    bt_props : dict
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary.

    """

    def __init__(self, name, extents, geom=False, **bt_props):
        super().__init__(name=name, geom=geom, **bt_props)
        self.extents = extents

    def create(self):
        # Physics
        body = bt.BulletRigidBodyNode(self.name + "_solid")
        self.bodies.append(body)
        self._set_properties(body)
        l, w, h, t = self.extents
        bottom = bt.BulletBoxShape(Vec3(l/2, w/2 - t, t/2))
        body.add_shape(bottom, TransformState.make_pos(Point3(0, 0, (t-h)/2)))
        side = bt.BulletBoxShape(Vec3(l/2, t/2, h/2))
        body.add_shape(side, TransformState.make_pos(Point3(0, (t-w)/2, 0)))
        body.add_shape(side, TransformState.make_pos(Point3(0, (w-t)/2, 0)))
        # Scene graph
        self.path = NodePath(body)
        # Geometry
        if self.geom:
            self.path.attach_new_node(
                self.make_geom(self.name + "_geom", self.extents))
        return self.path

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
