import math

import numpy as np
import panda3d.bullet as bt
import scipy.interpolate as ip
import scipy.optimize as opt
from panda3d.core import Quat, NodePath, Point3, TransformState, Vec3

from .base import PrimitiveBase
from .ball import Ball
from .cylinder import Cylinder
from .empty import Empty
from .tension_rope import _VisualRopeCallback


def get_xform_between_vectors(u, v):
    cross = u.cross(v)
    dot = u.dot(v)
    theta = math.atan2(cross.length(), dot)
    q = Quat(math.cos(theta/2), cross.normalized() * math.sin(theta/2))
    return TransformState.make_quat(q)


class _RopePulleyCallback:
    def __init__(self, components, hooks, constraints, rope_length, pulleys):
        self.comp1, self.comp2 = components
        self.hook1, self.hook2 = hooks
        self.slider1_cs = constraints[1]
        self.slider2_cs = constraints[4]
        self.hook1_cs = constraints[2]
        self.hook2_cs = constraints[5]
        self.rope_length = rope_length
        self.pulleys = pulleys
        # Useful values.
        self.dist_pulleys = sum(
            (c2 - c1).length() for c2, c1 in zip(pulleys[1:], pulleys[:-1])
        )
        self.max_dist = self.rope_length - self.dist_pulleys
        # Useful variables.
        self._dt = 0.  # time since previous update
        self._in_tension = False
        self._old_xforms = (self.hook1.get_net_transform(),
                            self.hook2.get_net_transform())

    def __call__(self, callback_data: bt.BulletTickCallbackData):
        # This callback may be called more often than the objects being
        # actually updated. Check if an update is necessary.
        stale = self._check_stale(callback_data)
        if stale:
            dt = self._dt + callback_data.get_timestep()
            self._dt = 0
        else:
            self._dt += callback_data.get_timestep()
            return
        slider1 = self.slider1_cs
        slider2 = self.slider2_cs
        loose_rope = self.loose_rope
        max_dist = self.max_dist
        # If in tension now
        if loose_rope <= 0:
            # If in tension before
            if self._in_tension:
                weight = self._get_weight_force()
                delta = dt * weight
                new_dist1 = slider1.get_upper_linear_limit() + delta
                new_dist2 = slider2.get_upper_linear_limit() - delta
                # Clamp values between hard limits.
                if new_dist1 < 0 or new_dist2 > max_dist:
                    new_dist1 = 0
                    new_dist2 = max_dist
                if new_dist2 < 0 or new_dist1 > max_dist:
                    new_dist2 = 0
                    new_dist1 = max_dist
            else:
                self._in_tension = True
                new_dist1 = self.vec1.length() + loose_rope/2
                new_dist2 = self.vec2.length() + loose_rope/2
            slider1.set_upper_linear_limit(new_dist1)
            slider2.set_upper_linear_limit(new_dist2)
        # If in tension before but not anymore
        elif self._in_tension:
            self._in_tension = False
            slider1.set_upper_linear_limit(self.max_dist)
            slider2.set_upper_linear_limit(self.max_dist)
        # If not in tension now or before, don't do anything.

    def check_physically_valid(self):
        return self.loose_rope >= 0

    def _check_stale(self, callback_data: bt.BulletTickCallbackData):
        # Check that objects' transforms have been updated.
        xforms = (self.hook1.get_net_transform(),
                  self.hook2.get_net_transform())
        stale = (self._old_xforms[0] != xforms[0] or
                 self._old_xforms[1] != xforms[1])
        self._old_xforms = xforms
        return stale

    @property
    def loose_rope(self):
        return self.max_dist - self.vec1.length() - self.vec2.length()

    @property
    def vec1(self):
        return self.hook1.get_pos() - self.pulleys[0]

    @property
    def vec2(self):
        return self.hook2.get_pos() - self.pulleys[-1]

    def _get_weight_force(self):
        gravity = 9.81
        mass1 = self.comp1.node().get_mass()
        mass2 = self.comp2.node().get_mass()
        return gravity * (mass1 - mass2) / (mass1 + mass2)


class _VisualRopePulleyCallback(_VisualRopeCallback):
    def __init__(self, name, parent, hooks, pulleys, rope_length, geom):
        self.pulleys = pulleys
        # Useful values.
        self.dist_pulleys = sum(
            (c2 - c1).length() for c2, c1 in zip(pulleys[1:], pulleys[:-1])
        )
        self.max_dist = rope_length - self.dist_pulleys
        super().__init__(name, parent, hooks, rope_length, geom)

    @property
    def loose_rope(self):
        return self.max_dist - self.vec1.length() - self.vec2.length()

    @property
    def vec1(self):
        return self.hook1.get_pos() - self.pulleys[0]

    @property
    def vec2(self):
        return self.hook2.get_pos() - self.pulleys[-1]

    def _get_rope_vertices(self):
        P0 = self.hook1.get_pos(self.parent)
        P1 = self.pulleys[0]
        Pn_1 = self.pulleys[-1]
        Pn = self.hook2.get_pos(self.parent)
        t = np.linspace(0, 1, self.n_vertices-2)
        loose_rope = max(0, self.loose_rope)
        vertices = [P0]
        for ti in t:
            p = P1 * (1-ti) + Pn_1 * ti
            p[2] -= loose_rope * .5 * math.sin(math.pi * ti)
            vertices.append(p)
        vertices.append(Pn)
        return vertices


class RopePulley(PrimitiveBase):
    """Create a rope-pulley system connecting two primitives.

    Parameters
    ----------
    name : string
      Name of the primitive.
    comp1_pos : (3,) float sequence
      Relative position of the hook on the first component.
    comp2_pos : (3,) float sequence
      Relative position of the hook on the second component.
    rope_length : float
      Length of the rope.
    pulleys : (n,3) float array
      (x, y, z) of each pulley.
    pulley_extents : float pair
      Radius and height of the cylinder.
    pulley_hpr : (3,) float sequence
      Orientation of all the pulleys.

    """

    def __init__(self, name, comp1_pos, comp2_pos, rope_length, pulleys,
                 pulley_extents, pulley_hpr):
        super().__init__(name)
        self.comp1_pos = Point3(*comp1_pos)
        self.comp2_pos = Point3(*comp2_pos)
        self.rope_length = rope_length
        self.pulleys = [Point3(*c) for c in pulleys]
        self.pulley_hpr = Vec3(*pulley_hpr)
        # Useful values.
        self.dist_pulleys = sum(
            (c2 - c1).length()
            for c2, c1 in zip(self.pulleys[1:], self.pulleys[:-1])
        )
        self.max_dist = self.rope_length - self.dist_pulleys
        # Hardcoded physical properties.
        self.hook_mass = 1e-2
        # self.max_slider_force = 1e6
        # Visual properties.
        self.pulley_extents = pulley_extents

    def _attach_objects(self, geom, phys, parent, world, components):
        hook1, cs1 = self._attach_pulley(components[0], self.comp1_pos,
                                         self.pulleys[0], parent, phys, world)
        hook2, cs2 = self._attach_pulley(components[1], self.comp2_pos,
                                         self.pulleys[-1], parent, phys, world)
        if phys:
            cb = _RopePulleyCallback(components, (hook1, hook2), cs1+cs2,
                                     self.rope_length, self.pulleys)
            self._attach(physics_callback=cb, world=world)
        if geom is not None:
            cb = _VisualRopePulleyCallback(
                self.name+"_rope", parent, (hook1, hook2), self.pulleys,
                self.rope_length, geom
            )
            self._attach(physics_callback=cb, world=world)

    def _attach_pulley(self, component, comp_coords, pulley_coords,
                       parent, phys, world):
        name = component.get_name()
        object_hook_coords = component.get_transform(
        ).get_mat().xform_point(comp_coords)
        # Each pulley connection is a combination of three constraints: One
        # pivot at the pulley, another at the component, and a slider between
        # them.
        # Pulley hook (can rotate on the base)
        pulley_hook = Ball(  # using Ball instead of Empty stabilizes it
            name + "_pulley-hook", self.pulley_extents[0], mass=self.hook_mass
        ).create(None, phys, parent, world)
        pulley_hook.set_pos(pulley_coords)
        # Object hook (can rotate on the object)
        object_hook = Empty(
            name + "_object-hook", mass=self.hook_mass
        ).create(None, phys, parent, world)
        object_hook.set_pos(object_hook_coords)
        # Physics
        if phys:
            # component.node().set_deactivation_enabled(False)
            # pulley_hook.node().set_deactivation_enabled(False)
            # object_hook.node().set_deactivation_enabled(False)
            pulley_hpr = self.pulley_hpr.normalized()
            # Constraints
            cs1 = bt.BulletHingeConstraint(
                pulley_hook.node(), Point3(0), pulley_hpr
            )
            x = Vec3.unit_x()  # slider axis is along the X-axis by default
            axis = object_hook.get_pos() - pulley_hook.get_pos()
            xform = get_xform_between_vectors(x, axis)
            cs2 = bt.BulletSliderConstraint(
                pulley_hook.node(), object_hook.node(), xform, xform, True
            )
            cs2.set_lower_linear_limit(0)
            cs2.set_upper_linear_limit(self.max_dist)
            # cs2.set_max_linear_motor_force(self.max_slider_force)
            cs3 = bt.BulletHingeConstraint(
                object_hook.node(), component.node(),
                Point3(0), comp_coords, pulley_hpr, pulley_hpr, True
            )
            self._attach(constraints=(cs1, cs2, cs3), world=world)
            return object_hook, (cs1, cs2, cs3)
        else:
            return object_hook, ()

    def create(self, geom, phys, parent=None, world=None, components=None):
        # Scene graph
        path = NodePath(self.name)
        self._attach(path, parent)
        # Components
        if components:
            self._attach_objects(geom, phys, path, world, components)
        # Geometry
        if geom is not None:
            pulley_hpr = self.pulley_hpr
            for i, coords in enumerate(self.pulleys):
                n_seg = 2**5 if geom == 'HD' else 2**4
                pulley = path.attach_new_node(
                    Cylinder.make_geom(
                        self.name + "_pulley_" + str(i) + "_geom",
                        self.pulley_extents, center=True, n_seg=n_seg
                    )
                )
                pulley.set_pos(coords)
                pulley.set_hpr(pulley_hpr)
        return path


class RopePulley2(PrimitiveBase):
    """Create a rope-pulley system connecting two primitives.

    Parameters
    ----------
    name : string
      Name of the primitive.
    comp1_pos : (3,) float sequence
      Relative position of the hook on the first component.
    comp2_pos : (3,) float sequence
      Relative position of the hook on the second component.
    rope_extents : float triplet
      Radius, total length and segment length of the rope.
    pulleys : (n,3) float array
      (x, y, z) of each pulley.
    pulley_extents : float pair
      Radius and height of the cylinder.

    """
    def __init__(self, name, comp1_pos, comp2_pos, rope_extents, pulleys,
                 pulley_extents):
        super().__init__(name)
        self.comp1_pos = Point3(*comp1_pos)
        self.comp2_pos = Point3(*comp2_pos)
        self.rope_radius, self.rope_length, self.seg_len = rope_extents
        self.seg_mass = .00004
        self.seg_angular_damping = .9
        self.pulleys = np.asarray(pulleys)
        self.pulley_radius, self.pulley_height = pulley_extents

    def create(self, geom, phys, parent=None, world=None, components=None):
        # Scene graph
        path = NodePath(self.name)
        # Physics
        bodies = []
        constraints = []
        # Pulleys
        pulley_shape = bt.BulletCylinderShape(self.pulley_radius,
                                              self.pulley_height)
        border_shape = bt.BulletCylinderShape(
            3*self.rope_radius+self.pulley_radius, self.rope_radius
        )
        border_pos = Point3(0, 0, self.pulley_height/2 + self.rope_radius/2)
        border_xform = TransformState.make_pos(border_pos)
        border_xform2 = TransformState.make_pos(-border_pos)
        pulley_hpr = self._get_pulley_hpr()
        for i, coords in enumerate(self.pulleys):
            name = self.name + "_pulley_{}_solid".format(i)
            body = bt.BulletRigidBodyNode(name)
            bodies.append(body)
            body.add_shape(pulley_shape)
            body.add_shape(border_shape, border_xform)
            body.add_shape(border_shape, border_xform2)
            pulley_path = NodePath(body)
            pulley_path.reparent_to(path)
            pulley_path.set_pos(*coords)
            pulley_path.set_hpr(pulley_hpr)
        # Rope
        rope_i = len(bodies)
        rope_points = self._get_rope_points(components)
        seg_shape = bt.BulletCapsuleShape(self.rope_radius, self.seg_len)
        seg_xform = TransformState.make_pos(Point3(0, 0, self.seg_len/2))
        for i, (a, b) in enumerate(zip(rope_points[:-1], rope_points[1:])):
            name = self.name + "_ropeseg_{}_solid".format(i)
            body = bt.BulletRigidBodyNode(name)
            bodies.append(body)
            body.add_shape(seg_shape, seg_xform)
            body.set_mass(self.seg_mass)
            body.set_angular_damping(self.seg_angular_damping)
            seg_path = NodePath(body)
            seg_path.reparent_to(path)
            seg_path.set_pos(*a)
            seg_path.look_at(*b)
            seg_path.set_hpr(seg_path, Vec3(90, 0, 90))
        cs_xform = Point3(0, 0, self.seg_len)
        for b1, b2 in zip(bodies[rope_i:-1], bodies[rope_i+1:]):
            cs = bt.BulletSphericalConstraint(b1, b2, cs_xform, 0)
            constraints.append(cs)
        if components:
            cs1 = bt.BulletSphericalConstraint(
                components[0].node(), bodies[rope_i], self.comp1_pos, 0
            )
            cs2 = bt.BulletSphericalConstraint(
                components[1].node(), bodies[-1], self.comp2_pos, cs_xform
            )
            constraints.extend([cs1, cs2])
        # Attach all
        self._attach(path, parent, bodies=bodies, constraints=constraints,
                     world=world)
        return path

    def _get_pulley_hpr(self):
        pulley_line = self.pulleys[-1] - self.pulleys[0]
        if pulley_line[0]:
            pulley_hpr = Vec3(0, 90, 0)
        else:
            pulley_hpr = Vec3(0, 0, 90)
        return pulley_hpr

    def _get_rope_points(self, components):
        # Compute init points
        pulleys = self.pulleys
        comp1, comp2 = components
        pos1 = comp1.get_net_transform().get_mat().xform_point(self.comp1_pos)
        pos2 = comp2.get_net_transform().get_mat().xform_point(self.comp2_pos)
        init_points = np.empty((2*len(pulleys)+2, 3))
        init_points[0] = pos1
        init_points[-1] = pos2
        shift = self.pulley_radius + 2*self.rope_radius
        pulley_dir = pulleys[1] - pulleys[0]
        shift_x = shift * np.sign(pulley_dir[0])
        shift_y = shift * np.sign(pulley_dir[1])
        init_points[1:2*len(pulleys):2] = pulleys + [-shift_x, -shift_y, shift]
        init_points[2:2*len(pulleys)+1:2] = pulleys + [shift_x, shift_y, shift]
        distances = np.linalg.norm(init_points[1:] - init_points[:-1], axis=1)
        residual = self.rope_length - distances.sum()
        if residual > 0:
            # print("r", residual)
            cat = self._solve_catenary_3d(init_points[2], init_points[3],
                                          distances[2]+residual)
            loose_points = cat(np.linspace(0, 1, 10)[1:-1])
            init_points = np.insert(init_points, 3, loose_points, axis=0)
        # Compute subdivided points
        distances = np.linalg.norm(init_points[1:] - init_points[:-1], axis=1)
        init_t = np.zeros(len(init_points))
        init_t[1:] = np.cumsum(distances)
        n_seg = int(init_t[-1] / self.seg_len)
        t = np.linspace(0, init_t[-1], n_seg)
        rope_points = np.column_stack([
            ip.interp1d(init_t, init_points[:, d])(t) for d in range(3)
        ])
        # print(rope_points)
        return rope_points

    def _solve_catenary_3d(self, p1, p2, s):
        h = math.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
        v = p2[2] - p1[2]
        rhs = math.sqrt(s**2 - v**2)
        sinh, arcsinh, cosh = np.sinh, np.arcsinh, np.cosh

        def f(x):
            return (2*x*sinh(h/(2*x)) - rhs) ** 2

        def fprime(x):
            return 2 * (2*sinh(h/(2*x)) - h*cosh(h/(2*x))/x) * (
                2*x*sinh(h/(2*x)) - rhs)
        x0_test = np.linspace(.01, s, 20)
        x0 = x0_test[np.argmin(f(x0_test))]
        a = opt.newton(f, x0=x0, fprime=fprime)
        # Compute the vertex coordinates
        x0 = a*arcsinh(v / (2*a*sinh(h/(2*a)))) - h/2
        z0 = p2[2] - a*cosh((h+x0)/a)

        def catenary(t):
            p = np.outer(t, (p2 - p1)) + p1
            p[:, 2] = a*cosh((t*h + x0) / a) + z0
            return p
        return catenary
