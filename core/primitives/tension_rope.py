import math

import numpy as np
import panda3d.bullet as bt
from panda3d.core import (LineSegs, Quat, NodePath, Point3,
                          TransformState, Vec3)

from .base import PrimitiveBase
from .ball import Ball
from .cylinder import Cylinder


class _VisualRopeCallback:
    def __init__(self, name, parent, hooks, rope_length, geom):
        self.name = name
        self.parent = parent
        self.hook1, self.hook2 = hooks
        self.rope_length = rope_length
        # Visual
        self.n_vertices = 15
        self.thickness = .001
        if geom == 'LD':
            self.rope = self._create_rope_ld()
            self._update_rope = self._update_rope_ld
        elif geom == 'HD':
            self.rope = self._create_rope_hd()
            self._update_rope = self._update_rope_hd
        # Useful variables.
        self._dt = 0.
        self._old_xforms = (self.hook1.get_net_transform(),
                            self.hook2.get_net_transform())

    def __call__(self, callback_data: bt.BulletTickCallbackData):
        if self._check_stale(callback_data):
            self._update_rope(self.rope)

    def check_physically_valid(self):
        return True

    @property
    def loose_rope(self):
        return 0

    def _check_stale(self, callback_data: bt.BulletTickCallbackData):
        # Check that objects' transforms have been updated.
        xforms = (self.hook1.get_net_transform(),
                  self.hook2.get_net_transform())
        stale = (self._old_xforms[0] != xforms[0] or
                 self._old_xforms[1] != xforms[1])
        self._old_xforms = xforms
        return stale

    def _create_rope_hd(self):
        thickness = self.thickness
        base_name = self.name
        rope = NodePath(base_name)
        for i in range(1, self.n_vertices):
            name = base_name + "_seg" + str(i)
            geom = Cylinder.make_geom(name, (thickness, 1), 4, False)
            geom.set_tag('anim_id', '')
            geom.set_tag('save_scale', '')
            rope.attach_new_node(geom)
        self._update_rope_hd(rope)
        rope.reparent_to(self.parent)
        return rope

    def _create_rope_ld(self):
        vertices = self._get_rope_vertices()
        ls = LineSegs(self.name)
        ls.set_color(0)
        vertiter = iter(vertices)
        ls.move_to(next(vertiter))
        for v in vertiter:
            ls.draw_to(v)
        rope = NodePath(ls.create(dynamic=True))
        self._rope_maker = ls
        rope.reparent_to(self.parent)
        return rope

    def _get_rope_vertices(self):
        P1 = self.hook1.get_pos(self.parent)
        P2 = self.hook2.get_pos(self.parent)
        t = np.linspace(0, 1, self.n_vertices)
        loose_rope = max(0, self.rope_length - (P1-P2).length())
        vertices = []
        for ti in t:
            p = P1 * (1-ti) + P2 * ti
            p[2] -= loose_rope * .5 * math.sin(math.pi * ti)
            vertices.append(p)
        return vertices

    def _update_rope_hd(self, rope):
        vertices = self._get_rope_vertices()
        # Update rope
        for i in range(len(vertices)-1):
            seg = rope.get_child(i)
            a = vertices[i]
            b = vertices[i+1]
            seg.set_pos(a)
            seg.set_scale(Vec3(1, 1, (b - a).length()))
            seg.look_at(b)
            seg.set_hpr(seg, Vec3(90, 0, 90))

    def _update_rope_ld(self, rope):
        vertices = self._get_rope_vertices()
        # Update rope
        ls = self._rope_maker
        for i, v in enumerate(vertices):
            ls.set_vertex(i, v)


def get_xform_between_vectors(u, v):
    cross = u.cross(v)
    dot = u.dot(v)
    theta = math.atan2(cross.length(), dot)
    q = Quat(math.cos(theta/2), cross.normalized() * math.sin(theta/2))
    return TransformState.make_quat(q)


class TensionRope(PrimitiveBase):
    """Create a rope in tension between two primitives.

    Parameters
    ----------
    name : string
      Name of the primitive.
    comp1_pos : (3,) float sequence
      Relative position of the hook on the first component.
    comp2_pos : (3,) float sequence
      Relative position of the hook on the second component.
    pivot_hpr: (3,) float sequence
      Orientation of the pivot constraints.
    hook_radius : float, optional
      Radius of the visual hook.
    loose_rope : float, optional
      Additional loose rope.

    """

    def __init__(self, name, comp1_pos, comp2_pos, pivot_hpr, hook_radius=.01,
                 loose_rope=0):
        super().__init__(name)
        self.comp1_pos = Point3(*comp1_pos)
        self.comp2_pos = Point3(*comp2_pos)
        self.pivot_hpr = Vec3(*pivot_hpr)
        self.loose_rope = loose_rope
        # Hardcoded physical properties.
        self.hook_mass = 5e-3
        self.max_slider_force = 1e6
        # Visual properties.
        self.hook_radius = hook_radius

    def create(self, geom, phys, parent=None, world=None, components=None):
        # Scene graph
        path = NodePath(self.name)
        self._attach(path, parent)
        # Components
        comp1, comp2 = components
        name1 = comp1.get_name()
        name2 = comp2.get_name()
        # The rope connection is a combination of three constraints: one pivot
        # at each component and a slider between them.
        # Hooks
        hook1 = Ball(  # using Ball instead of Empty stabilizes it
            name1 + "_hook", self.hook_radius, mass=self.hook_mass
        ).create(geom, phys, comp1, world)
        hook1.set_pos(self.comp1_pos)
        hook2 = Ball(  # using Ball instead of Empty stabilizes it
            name2 + "_hook", self.hook_radius, mass=self.hook_mass
        ).create(geom, phys, comp2, world)
        hook2.set_pos(self.comp2_pos)
        length = (
            hook1.get_net_transform().pos - hook2.get_net_transform().pos
        ).length() + self.loose_rope
        # Physics
        if phys:
            # comp1.node().set_deactivation_enabled(False)
            # comp2.node().set_deactivation_enabled(False)
            # hook1.node().set_deactivation_enabled(False)
            # hook2.node().set_deactivation_enabled(False)
            pivot_hpr = self.pivot_hpr.normalized()
            # Constraints
            cs1 = bt.BulletHingeConstraint(
                hook1.node(), comp1.node(),
                Point3(0), self.comp1_pos, pivot_hpr, pivot_hpr, True
            )
            cs2 = bt.BulletHingeConstraint(
                hook2.node(), comp2.node(),
                Point3(0), self.comp2_pos, pivot_hpr, pivot_hpr, True
            )
            x = Vec3.unit_x()  # slider axis is along the X-axis by default
            axis = hook2.get_pos(hook1)
            xform = get_xform_between_vectors(x, axis)
            cs3 = bt.BulletSliderConstraint(
                hook1.node(), hook2.node(), xform, xform, True
            )
            cs3.set_lower_linear_limit(0)
            cs3.set_upper_linear_limit(length)
            self._attach(constraints=(cs1, cs2, cs3), world=world)
        if geom is not None:
            # Rope
            cb = _VisualRopeCallback(self.name, path, (hook1, hook2),
                                     length, geom)
            self._attach(physics_callback=cb, world=world)
        return path
