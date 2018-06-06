import math
import pickle
import subprocess
from collections import namedtuple

import chaospy as cp
import numpy as np
from panda3d.core import NodePath, Point3, TransformState, Vec3

import core.primitives as prim
import xp.adventure.config as cfg
import xp.causal as causal
from core.export import VectorFile
from xp.scenarios import CausalGraphTerminationCondition, Samplable, Scenario

Scene = namedtuple('Scene', ['graph', 'world'])


def init_scene():
    """Initialize the Panda3D scene."""
    graph = NodePath("scene")
    world = prim.World()
    world.set_gravity(cfg.GRAVITY)
    return Scene(graph, world)


class RollingOn:
    def __init__(self, rolling, support, world):
        self.rolling = rolling
        self.support = support
        self.world = world
        self.start_angle = None

    def __call__(self):
        contact = self.world.contact_test_pair(
            self.rolling.node(), self.support.node()
        ).get_num_contacts()
        if contact:
            if self.start_angle is None:
                self.start_angle = self.rolling.get_quat().get_angle()
        else:
            self.start_angle = None
            return False
        angle = abs(self.rolling.get_quat().get_angle() - self.start_angle)
        return angle >= cfg.ROLLING_ANGLE and contact


class Contact:
    def __init__(self, first, second, world):
        self.first = first
        self.second = second
        self.world = world

    def __call__(self):
        contact = self.world.contact_test_pair(
            self.first.node(), self.second.node()
        ).get_num_contacts()
        return contact


class NoContact:
    def __init__(self, first, second, world):
        self.first = first
        self.second = second
        self.world = world

    def __call__(self):
        contact = self.world.contact_test_pair(
            self.first.node(), self.second.node()
        ).get_num_contacts()
        return not contact


class Inclusion:
    def __init__(self, inside, outside):
        self.inside = inside
        self.outside = outside

    def __call__(self):
        in_bounds = self.inside.node().get_shape_bounds()
        out_bounds = self.outside.node().get_shape_bounds()
        in_center = in_bounds.get_center() + self.inside.get_pos()
        out_center = out_bounds.get_center() + self.outside.get_pos()
        include = ((in_center - out_center).length()
                   + in_bounds.get_radius()) <= out_bounds.get_radius()
        return include


class Toppling:
    def __init__(self, body, angle):
        self.body = body
        self.angle = angle
        self.start_angle = body.get_r()

    def __call__(self):
        return abs(self.body.get_r() - self.start_angle) >= self.angle + 1


class Pivoting:
    def __init__(self, body):
        self.body = body

    def __call__(self):
        angvel = self.body.node().get_angular_velocity().length_squared()
        return angvel > cfg.PIVOTING_ANGULAR_VELOCITY


class Stopping:
    def __init__(self, body):
        self.body = body

    def __call__(self):
        linvel = self.body.node().get_linear_velocity().length()
        angvel = self.body.node().get_angular_velocity().length()
        return (linvel < cfg.STOPPING_LINEAR_VELOCITY
                and angvel < cfg.STOPPING_ANGULAR_VELOCITY)


class Falling:
    def __init__(self, body):
        self.body = body

    def __call__(self):
        linvel = self.body.node().get_linear_velocity()[2]
        return linvel < cfg.FALLING_LINEAR_VELOCITY


class Rising:
    def __init__(self, body):
        self.body = body

    def __call__(self):
        linvel = self.body.node().get_linear_velocity()[2]
        return linvel > cfg.RISING_LINEAR_VELOCITY


class Dummy:
    def __call__(self):
        return True


class StateObserver:
    """Keeps track of the full state of each non-static object in the scene."""
    def __init__(self, scene):
        self.graph_root = scene.graph
        self.paths = []
        self.states = dict()
        # Find and add nodes that are already tagged.
        for path in self.graph_root.find_all_matches("**/=anim_id"):
            anim_id = str(path.get_key())
            path.set_tag('anim_id', anim_id)
            self.states[anim_id] = []
            self.paths.append(path)
        # Find and tag any GeomNode child of a non-static BulletRigidBodyNode.
        for body in scene.world.get_rigid_bodies():
            if not body.is_static() and body.get_num_children():
                child = body.get_child(0)
                if child.is_geom_node():
                    path = NodePath.any_path(child)
                    anim_id = str(path.get_key())
                    path.set_tag('anim_id', anim_id)
                    self.states[anim_id] = []
                    self.paths.append(path)

    def __call__(self, time):
        for path in self.paths:
            anim_id = path.get_tag('anim_id')
            x, y, z = path.get_pos(self.graph_root)
            w, i, j, k = path.get_quat(self.graph_root)
            if path.has_tag('save_scale'):
                sx, sy, sz = path.get_scale(self.graph_root)
                state = [time, x, y, z, w, i, j, k, sx, sy, sz]
            else:
                state = [time, x, y, z, w, i, j, k]
            self.states[anim_id].append(state)

    def export(self, filename):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.states, f)


class TeapotAdventure(Samplable, Scenario):
    """Scenario

    Parameters
    ----------
    sample : (-,) sequence
      []

    """
    def __init__(self, sample, make_geom=False, graph_view=False, **kwargs):
        self._scene = self.init_scene(sample, make_geom)
        self.causal_graph = self.init_causal_graph(self._scene,
                                                   verbose=make_geom)
        if graph_view:
            self.graph_view = causal.CausalGraphViewer(self.causal_graph.root)
        # LEGACY
        self.world = self._scene.world
        self.scene = self._scene.graph
        self.terminate = CausalGraphTerminationCondition(self.causal_graph)

    def check_physically_valid(self):
        return self._check_physically_valid_scene(self._scene)

    @classmethod
    def check_physically_valid_sample(cls, sample):
        scene = cls.init_scene(sample, make_geom=False)
        return cls._check_physically_valid_scene(scene)

    @staticmethod
    def _check_physically_valid_scene(scene):
        scene_graph = scene.graph
        # Check pulleys
        world = scene.world
        if not all(pulley_cb.__self__.check_physically_valid()
                   for pulley_cb in world._callbacks):
            return False
        # Check unwanted collisions.
        top_track = scene_graph.find("top_track_solid").node()
        middle_track = scene_graph.find("middle_track_solid").node()
        nail = scene_graph.find("nail_solid").node()
        gate = scene_graph.find("gate_solid").node()
        right_track1 = scene_graph.find("right_track1_solid").node()
        right_track2 = scene_graph.find("right_track2_solid").node()
        right_track2_blocker1 = scene_graph.find("right_track2_blocker1_solid"
                                                 ).node()
        right_track2_blocker2 = scene_graph.find("right_track2_blocker2_solid"
                                                 ).node()
        right_track3 = scene_graph.find("right_track3_solid").node()
        right_track4 = scene_graph.find("right_track4_solid").node()
        right_weight = scene_graph.find("right_weight_solid").node()
        left_track1 = scene_graph.find("left_track1_solid").node()
        left_track2 = scene_graph.find("left_track2_solid").node()
        left_track3 = scene_graph.find("left_track3_solid").node()
        left_track4 = scene_graph.find("left_track4_solid").node()
        left_weight = scene_graph.find("left_weight_solid").node()
        bridge = scene_graph.find("bridge_solid").node()
        bottom_goblet = scene_graph.find("bottom_goblet_solid").node()
        teapot_base = scene_graph.find("teapot_base_solid").node()
        # Enable collisions for static objects
        static = (
            top_track, middle_track,
            right_track1, right_track2, right_track3, right_track4,
            right_track2_blocker1, right_track2_blocker2,
            left_track1, left_track2, left_track3, left_track4,
            bottom_goblet, teapot_base
        )
        for body in static:
            body.set_static(False)
            body.set_active(True)
        test_pairs = [
            (top_track, gate),
            (top_track, middle_track),
            (top_track, left_track1),
            (nail, middle_track),
            (left_weight, left_track1),
            (left_weight, left_track2),
            (left_weight, left_track4),
            (left_track2, left_track1),
            (left_track3, left_track2),
            (left_track4, left_track3),
            (left_track4, bottom_goblet),
            (left_track4, bridge),
            (bridge, bottom_goblet),
            (bridge, teapot_base),
            (right_weight, right_track1),
            (right_weight, right_track2),
            (right_weight, right_track4),
            (right_weight, right_track2_blocker1),
            (right_weight, right_track2_blocker2),
            (right_track2, right_track1),
            (right_track3, right_track2),
            (right_track4, right_track3),
            (right_track4, teapot_base),
        ]
        valid = not any(world.contact_test_pair(a, b).get_num_contacts()
                        for a, b in test_pairs)
        TransformState.garbage_collect()
        return valid

    @staticmethod
    def get_distribution():
        distributions = [
            cp.Truncnorm(xmin, xmax, (xmin+xmax)/2, (xmax-xmin)/4)
            for xmin, xmax in cfg.SCENARIO_PARAMETERS_BOUNDS
        ]
        return cp.J(*distributions)

    @classmethod
    def get_physical_validity_constraint(cls, sample):
        score = 0
        scene = cls.init_scene(sample, make_geom=False)
        scene_graph = scene.graph
        # Get pulleys' score
        world = scene.world
        for pulley_cb in world._callbacks:
            score += min(0, pulley_cb.__self__._get_loose_rope_length())
        # Check unwanted collisions.
        top_track = scene_graph.find("top_track_solid").node()
        middle_track = scene_graph.find("middle_track_solid").node()
        nail = scene_graph.find("nail_solid").node()
        gate = scene_graph.find("gate_solid").node()
        right_track1 = scene_graph.find("right_track1_solid").node()
        right_track2 = scene_graph.find("right_track2_solid").node()
        right_track2_blocker1 = scene_graph.find("right_track2_blocker1_solid"
                                                 ).node()
        right_track2_blocker2 = scene_graph.find("right_track2_blocker2_solid"
                                                 ).node()
        right_track3 = scene_graph.find("right_track3_solid").node()
        right_track4 = scene_graph.find("right_track4_solid").node()
        right_weight = scene_graph.find("right_weight_solid").node()
        left_track1 = scene_graph.find("left_track1_solid").node()
        left_track2 = scene_graph.find("left_track2_solid").node()
        left_track3 = scene_graph.find("left_track3_solid").node()
        left_track4 = scene_graph.find("left_track4_solid").node()
        left_weight = scene_graph.find("left_weight_solid").node()
        bridge = scene_graph.find("bridge_solid").node()
        bottom_goblet = scene_graph.find("bottom_goblet_solid").node()
        teapot_base = scene_graph.find("teapot_base_solid").node()
        # Enable collisions for static objects
        static = (
            top_track, middle_track,
            right_track1, right_track2, right_track3, right_track4,
            right_track2_blocker1, right_track2_blocker2,
            left_track1, left_track2, left_track3, left_track4,
            bottom_goblet, teapot_base
        )
        for body in static:
            body.set_static(False)
            body.set_active(True)
        test_pairs = [
            (top_track, gate),
            (top_track, middle_track),
            (top_track, left_track1),
            (nail, middle_track),
            (left_weight, left_track1),
            (left_weight, left_track2),
            (left_weight, left_track4),
            (left_track2, left_track1),
            (left_track3, left_track2),
            (left_track4, left_track3),
            (left_track4, bottom_goblet),
            (left_track4, bridge),
            (bridge, bottom_goblet),
            (bridge, teapot_base),
            (right_weight, right_track1),
            (right_weight, right_track2),
            (right_weight, right_track4),
            (right_weight, right_track2_blocker1),
            (right_weight, right_track2_blocker2),
            (right_track2, right_track1),
            (right_track3, right_track2),
            (right_track4, right_track3),
            (right_track4, teapot_base),
        ]
        for a, b in test_pairs:
            result = world.contact_test_pair(a, b)
            if result.get_num_contacts():
                penetration = 0
                for contact in result.get_contacts():
                    mpoint = contact.get_manifold_point()
                    penetration += mpoint.get_distance()
                score += penetration
        TransformState.garbage_collect()
        return score

    def export_scene_to_egg(self, filename):
        if filename[-4:] == ".egg":
            filename = filename[:-4]
        self._scene.graph.write_bam_file(filename + ".bam")
        subprocess.run(["bam2egg", "-o", filename + ".egg", filename + ".bam"])

    @classmethod
    def export_scene_to_pdf(cls, filename, sample, sheetsize):
        coords = []
        sizes = []
        shapes = []

        def add_rect(name, dims):
            pos, hpr = cls.sample2coords(sample, name)
            coords.append([pos.x, pos.z, hpr.z])
            sizes.append([dims[0], dims[2]])
            shapes.append('rect')

        def add_circles(name, radius):
            pos = cls.sample2coords(sample, name)
            for p in pos:
                coords.append([p.x, p.z, 0])
                sizes.append([radius, 0])
                shapes.append('circ')

        def add_goblet(name):
            pos, hpr = cls.sample2coords(sample, name)
            angle = math.radians(hpr.z)
            coords.append([pos.x + math.sin(angle)*cfg.GOBLET_HEIGHT/2,
                           pos.z + math.cos(angle)*cfg.GOBLET_HEIGHT/2,
                           hpr.z])
            sizes.append([cfg.GOBLET_R1*2, cfg.GOBLET_HEIGHT])
            shapes.append('rect')

        add_rect("top_track", cfg.TOP_TRACK_LWHT)
        add_rect("middle_track", cfg.SHORT_TRACK_LWHT)
        add_rect("left_track1", cfg.LONG_TRACK_LWHT)
        add_rect("left_track2", cfg.SHORT_TRACK_LWHT)
        add_rect("left_track3", cfg.SHORT_TRACK_LWHT)
        add_rect("left_track3_blocker", cfg.FLAT_SUPPORT_LWH)
        add_rect("left_track4", cfg.LONG_TRACK_LWHT)
        add_rect("left_track4_blocker", cfg.FLAT_SUPPORT_LWH)
        add_rect("right_track1", cfg.LONG_TRACK_LWHT)
        add_rect("right_track2", cfg.SHORT_TRACK_LWHT)
        add_rect("right_track3", cfg.SHORT_TRACK_LWHT)
        add_rect("right_track3_blocker", cfg.FLAT_SUPPORT_LWH)
        add_rect("right_track4", cfg.SHORT_TRACK_LWHT)
        add_rect("right_track4_blocker", cfg.FLAT_SUPPORT_LWH)
        add_rect("gate", cfg.PLANK_LWH)
        add_rect("top_weight_guide", cfg.FLAT_SUPPORT_LWH)
        add_rect("top_weight_support", cfg.TINY_TRACK_LWH)
        add_rect("sliding_plank", (cfg.PLANK_LWH[0], 0, cfg.PLANK_LWH[1]))
        add_rect("nail", cfg.NAIL_LEVER_LWH)
        add_rect("left_weight", cfg.QUAD_PLANK_LWH)
        add_rect("left_weight_support", cfg.FLAT_SUPPORT_LWH)
        add_rect("right_weight", (cfg.RIGHT_WEIGHT_HEIGHT,
                                  0, 2*cfg.RIGHT_WEIGHT_RADIUS))
        add_rect("right_weight_support", cfg.FLAT_SUPPORT_LWH)
        add_rect("bridge", cfg.TINY_TRACK_LWH)
        pos = cls.sample2coords(sample, "bridge")[0]
        coords.append([pos.x + 0.015, pos.z, 0])
        sizes.append([.005, 0])
        shapes.append('circ')
        add_circles("top_pulley", .005)
        add_circles("left_pulley", .005)
        add_circles("right_pulley", .005)
        add_goblet("top_goblet")
        add_goblet("bottom_goblet")
        add_goblet("teapot_base")

        coords = np.asarray(coords)
        sizes = np.asarray(sizes)
        rects = np.array([shape == 'rect' for shape in shapes])
        circles = np.array([shape == 'circ' for shape in shapes])

        xy = coords[:, :2] * 100
        xy[:, 1] *= -1
        xy = xy - (xy.min(axis=0) + xy.max(axis=0))/2 + np.asarray(sheetsize)/2
        a = coords[:, 2]
        sizes *= 100

        vec = VectorFile(filename, sheetsize)
        vec.add_rectangles(xy[rects], a[rects], sizes[rects])
        vec.add_circles(xy[circles], sizes[circles, 0])
        vec.save()

    @staticmethod
    def init_causal_graph(scene, verbose=False):
        scene_graph = scene.graph
        world = scene.world
        ball1 = scene_graph.find("ball1_solid")
        ball2 = scene_graph.find("ball2_solid")
        top_track = scene_graph.find("top_track_solid")
        middle_track = scene_graph.find("middle_track_solid")
        nail = scene_graph.find("nail_solid")
        gate = scene_graph.find("gate_solid")
        top_goblet = scene_graph.find("top_goblet_solid")
        right_track1 = scene_graph.find("right_track1_solid")
        right_track2 = scene_graph.find("right_track2_solid")
        right_track3 = scene_graph.find("right_track3_solid")
        right_track4 = scene_graph.find("right_track4_solid")
        right_weight = scene_graph.find("right_weight_solid")
        left_track1 = scene_graph.find("left_track1_solid")
        left_track2 = scene_graph.find("left_track2_solid")
        left_track3 = scene_graph.find("left_track3_solid")
        left_track4 = scene_graph.find("left_track4_solid")
        left_weight = scene_graph.find("left_weight_solid")
        bridge = scene_graph.find("bridge_solid")
        teapot_base = scene_graph.find("teapot_base_solid")
        teapot_lid = scene_graph.find("teapot_lid_solid")

        ball2_rolls_on_top_track = causal.Event(
            "ball2_rolls_on_top_track",
            RollingOn(ball2, top_track, world),
            None,
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_falls_on_middle_track = causal.Event(
            "ball2_falls_on_middle_track",
            Contact(ball2, middle_track, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_hits_nail = causal.Event(
            "ball2_hits_nail",
            Contact(ball2, nail, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_falls_on_right_track1 = causal.Event(
            "ball2_falls_on_right_track1",
            Contact(ball2, right_track1, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_hits_right_weight = causal.Event(
            "ball2_hits_right_weight",
            Contact(ball2, right_weight, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_falls_on_right_track2 = causal.Event(
            "ball2_falls_on_right_track2",
            Contact(ball2, right_track2, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_falls_on_right_track3 = causal.Event(
            "ball2_falls_on_right_track3",
            Contact(ball2, right_track3, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_falls_on_right_track4 = causal.Event(
            "ball2_falls_on_right_track4",
            Contact(ball2, right_track4, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_enters_teapot = causal.Event(
            "ball2_enters_teapot",
            Inclusion(ball2, teapot_base),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        right_weight_falls = causal.Event(
            "right_weight_falls",
            Falling(right_weight),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        brige_pivots = causal.Event(
            "brige_pivots",
            Pivoting(bridge),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        gate_falls = causal.Event(
            "gate_falls",
            Falling(gate),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        top_goblet_rises = causal.Event(
            "top_goblet_rises",
            Rising(top_goblet),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_rolls_on_top_track = causal.Event(
            "ball1_rolls_on_top_track",
            RollingOn(ball1, top_track, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_hits_gate = causal.Event(
            "ball1_hits_gate",
            Contact(ball1, gate, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_falls_on_left_track1 = causal.Event(
            "ball1_falls_on_left_track1",
            Contact(ball1, left_track1, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_hits_left_weight = causal.Event(
            "ball1_hits_left_weight",
            Contact(ball1, left_weight, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        left_weight_falls = causal.Event(
            "left_weight_falls",
            Falling(left_weight),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        teapot_lid_rises = causal.Event(
            "teapot_lid_rises",
            Rising(teapot_lid),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_falls_on_left_track2 = causal.Event(
            "ball1_falls_on_left_track2",
            Contact(ball1, left_track2, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_falls_on_left_track3 = causal.Event(
            "ball1_falls_on_left_track3",
            Contact(ball1, left_track3, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_falls_on_left_track4 = causal.Event(
            "ball1_falls_on_left_track4",
            Contact(ball1, left_track4, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_falls_on_bridge = causal.Event(
            "ball1_falls_on_bridge",
            Contact(ball1, bridge, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_enters_teapot = causal.Event(
            "ball1_enters_teapot",
            Inclusion(ball1, teapot_base),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball1_stops = causal.Event(
            "ball1_stops",
            Stopping(ball1),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball2_stops = causal.Event(
            "ball2_stops",
            Stopping(ball2),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        end = causal.Event("end", Dummy(), causal.AllBefore(), None,
                           verbose=verbose)

        # First branch
        causal.connect(ball2_rolls_on_top_track, ball2_falls_on_middle_track)
        causal.connect(ball2_falls_on_middle_track, ball2_hits_nail)
        causal.connect(ball2_hits_nail, ball2_falls_on_right_track1)
        causal.connect(ball2_falls_on_right_track1, ball2_hits_right_weight)
        causal.connect(ball2_hits_right_weight, ball2_falls_on_right_track2)
        causal.connect(ball2_falls_on_right_track2,
                       ball2_falls_on_right_track3)
        causal.connect(ball2_falls_on_right_track3,
                       ball2_falls_on_right_track4)
        causal.connect(ball2_falls_on_right_track4, ball2_enters_teapot)
        causal.connect(ball2_enters_teapot, ball2_stops)
        causal.connect(ball2_stops, end)
        # Second branch
        causal.connect(ball2_hits_nail, gate_falls)
        causal.connect(gate_falls, top_goblet_rises)
        causal.connect(top_goblet_rises, ball1_rolls_on_top_track)
        causal.connect(ball1_rolls_on_top_track, ball1_hits_gate)
        causal.connect(ball1_hits_gate, ball1_falls_on_left_track1)
        causal.connect(ball1_falls_on_left_track1, ball1_hits_left_weight)
        causal.connect(ball1_hits_left_weight, ball1_falls_on_left_track2)
        causal.connect(ball1_falls_on_left_track2, ball1_falls_on_left_track3)
        causal.connect(ball1_falls_on_left_track3, ball1_falls_on_left_track4)
        causal.connect(ball1_falls_on_left_track4, ball1_falls_on_bridge)
        causal.connect(ball1_falls_on_bridge, ball1_enters_teapot)
        causal.connect(ball1_enters_teapot, ball1_stops)
        causal.connect(ball1_stops, end)
        # Sub-branch 1
        causal.connect(ball2_hits_right_weight, right_weight_falls)
        causal.connect(right_weight_falls, brige_pivots)
        causal.connect(brige_pivots, ball1_falls_on_bridge)
        # Sub-branch 2
        causal.connect(ball1_hits_left_weight, left_weight_falls)
        causal.connect(left_weight_falls, teapot_lid_rises)
        causal.connect(teapot_lid_rises, ball2_enters_teapot)

        graph = causal.CausalGraphTraverser(
            root=ball2_rolls_on_top_track, verbose=verbose
        )
        return graph

    @classmethod
    def init_scene(cls, sample, make_geom=False):
        scene = init_scene()

        ball1 = prim.Ball(
            "ball1",
            cfg.BALL_RADIUS,
            geom=make_geom,
            mass=cfg.BALL_MASS,
            friction=cfg.BALL_FRICTION
        )
        ball1.create().set_pos_hpr(
            *cls.sample2coords(sample, "ball1")
        )
        ball1.attach_to(scene.graph, scene.world)

        ball2 = prim.Ball(
            "ball2",
            cfg.BALL_RADIUS,
            geom=make_geom,
            mass=cfg.BALL_MASS
        )
        ball2.create().set_pos_hpr(
            *cls.sample2coords(sample, "ball2")
        )
        ball2.attach_to(scene.graph, scene.world)

        top_track = prim.Track(
            "top_track",
            cfg.TOP_TRACK_LWHT,
            geom=make_geom,
            friction=cfg.TOP_TRACK_FRICTION
        )
        top_track.create().set_pos_hpr(
            *cls.sample2coords(sample, "top_track")
        )
        top_track.attach_to(scene.graph, scene.world)

        middle_track = prim.Track(
            "middle_track",
            cfg.SHORT_TRACK_LWHT,
            geom=make_geom
        )
        middle_track.create().set_pos_hpr(
            *cls.sample2coords(sample, "middle_track")
        )
        middle_track.attach_to(scene.graph, scene.world)

        left_track1 = prim.Track(
            "left_track1",
            cfg.LONG_TRACK_LWHT,
            geom=make_geom
        )
        left_track1.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_track1")
        )
        left_track1.attach_to(scene.graph, scene.world)

        left_track2 = prim.Track(
            "left_track2",
            cfg.SHORT_TRACK_LWHT,
            geom=make_geom
        )
        left_track2.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_track2")
        )
        left_track2.attach_to(scene.graph, scene.world)

        left_track3 = prim.Track(
            "left_track3",
            cfg.SHORT_TRACK_LWHT,
            geom=make_geom
        )
        left_track3.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_track3")
        )
        left_track3.attach_to(scene.graph, scene.world)

        left_track3_blocker = prim.Box(
            "left_track3_blocker",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        left_track3_blocker.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_track3_blocker")
        )
        left_track3_blocker.attach_to(scene.graph, scene.world)

        left_track4 = prim.Track(
            "left_track4",
            cfg.LONG_TRACK_LWHT,
            geom=make_geom
        )
        left_track4.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_track4")
        )
        left_track4.attach_to(scene.graph, scene.world)

        left_track4_blocker = prim.Box(
            "left_track4_blocker",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        left_track4_blocker.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_track4_blocker")
        )
        left_track4_blocker.attach_to(scene.graph, scene.world)

        right_track1 = prim.Track(
            "right_track1",
            cfg.LONG_TRACK_LWHT,
            geom=make_geom
        )
        right_track1.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track1")
        )
        right_track1.attach_to(scene.graph, scene.world)

        right_track2 = prim.Track(
            "right_track2",
            cfg.SHORT_TRACK_LWHT,
            geom=make_geom
        )
        right_track2.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track2")
        )
        right_track2.attach_to(scene.graph, scene.world)

        right_track2_blocker1 = prim.Box(
            "right_track2_blocker1",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        right_track2_blocker1.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track2_blocker1")
        )
        right_track2_blocker1.attach_to(scene.graph, scene.world)

        right_track2_blocker2 = prim.Box(
            "right_track2_blocker2",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        right_track2_blocker2.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track2_blocker2")
        )
        right_track2_blocker2.attach_to(scene.graph, scene.world)

        right_track3 = prim.Track(
            "right_track3",
            cfg.SHORT_TRACK_LWHT,
            geom=make_geom
        )
        right_track3.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track3")
        )
        right_track3.attach_to(scene.graph, scene.world)

        right_track3_blocker = prim.Box(
            "right_track3_blocker",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        right_track3_blocker.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track3_blocker")
        )
        right_track3_blocker.attach_to(scene.graph, scene.world)

        right_track4 = prim.Track(
            "right_track4",
            cfg.SHORT_TRACK_LWHT,
            geom=make_geom
        )
        right_track4.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track4")
        )
        right_track4.attach_to(scene.graph, scene.world)

        right_track4_blocker = prim.Box(
            "right_track4_blocker",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        right_track4_blocker.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_track4_blocker")
        )
        right_track4_blocker.attach_to(scene.graph, scene.world)

        top_weight_support = prim.Box(
            "top_weight_support",
            cfg.TINY_TRACK_LWH,
            geom=make_geom
        )
        top_weight_support.create().set_pos_hpr(
            *cls.sample2coords(sample, "top_weight_support")
        )
        top_weight_support.attach_to(scene.graph, scene.world)

        top_weight_guide = prim.Box(
            "top_weight_guide",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        top_weight_guide.create().set_pos_hpr(
            *cls.sample2coords(sample, "top_weight_guide")
        )
        top_weight_guide.attach_to(scene.graph, scene.world)

        left_weight_support = prim.Box(
            "left_weight_support",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        left_weight_support.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_weight_support")
        )
        left_weight_support.attach_to(scene.graph, scene.world)

        right_weight_support = prim.Box(
            "right_weight_support",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        right_weight_support.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_weight_support")
        )
        right_weight_support.attach_to(scene.graph, scene.world)

        top_goblet = prim.Goblet(
            "top_goblet",
            (cfg.GOBLET_HEIGHT, cfg.GOBLET_R1, cfg.GOBLET_R2),
            geom=make_geom,
            mass=cfg.GOBLET_MASS,
            friction=cfg.GOBLET_FRICTION,
            angular_damping=cfg.GOBLET_ANGULAR_DAMPING
        )
        top_goblet.create().set_pos_hpr(
            *cls.sample2coords(sample, "top_goblet")
        )
        top_goblet.attach_to(scene.graph, scene.world)

        bottom_goblet = prim.Goblet(
            "bottom_goblet",
            (cfg.GOBLET_HEIGHT, cfg.GOBLET_R1, cfg.GOBLET_R2),
            geom=make_geom
        )
        bottom_goblet.create().set_pos_hpr(
            *cls.sample2coords(sample, "bottom_goblet")
        )
        bottom_goblet.attach_to(scene.graph, scene.world)

        sliding_plank = prim.Box(
            "sliding_plank",
            cfg.PLANK_LWH,
            geom=make_geom,
            mass=cfg.PLANK_MASS
        )
        sliding_plank.create().set_pos_hpr(
            *cls.sample2coords(sample, "sliding_plank")
        )
        sliding_plank.attach_to(scene.graph, scene.world)

        gate = prim.Box(
            "gate",
            cfg.PLANK_LWH,
            geom=make_geom,
            mass=cfg.PLANK_MASS
        )
        gate.create().set_pos_hpr(
            *cls.sample2coords(sample, "gate")
        )
        gate.attach_to(scene.graph, scene.world)

        left_weight = prim.Box(
            "left_weight",
            cfg.QUAD_PLANK_LWH,
            geom=make_geom,
            mass=4*cfg.PLANK_MASS,
        )
        left_weight.create().set_pos_hpr(
            *cls.sample2coords(sample, "left_weight")
        )
        left_weight.attach_to(scene.graph, scene.world)

        right_weight = prim.Cylinder(
            "right_weight",
            (cfg.RIGHT_WEIGHT_RADIUS, cfg.RIGHT_WEIGHT_HEIGHT),
            geom=make_geom,
            mass=cfg.RIGHT_WEIGHT_MASS
        )
        right_weight.create().set_pos_hpr(
            *cls.sample2coords(sample, "right_weight")
        )
        right_weight.attach_to(scene.graph, scene.world)

        bridge = prim.Box(
            "bridge",
            cfg.TINY_TRACK_LWH,
            geom=make_geom,
            mass=cfg.TINY_TRACK_MASS
        )
        bridge.create().set_pos_hpr(
            *cls.sample2coords(sample, "bridge")
        )
        bridge.attach_to(scene.graph, scene.world)

        teapot_base = prim.Goblet(
            "teapot_base",
            (cfg.GOBLET_HEIGHT, cfg.GOBLET_R1, cfg.GOBLET_R2),
            geom=make_geom,
            friction=cfg.TEAPOT_FRICTION
        )
        teapot_base.create().set_pos_hpr(
            *cls.sample2coords(sample, "teapot_base")
        )
        teapot_base.attach_to(scene.graph, scene.world)

        teapot_lid = prim.Cylinder(
            "teapot_lid",
            (cfg.TEAPOT_LID_RADIUS, cfg.TEAPOT_LID_HEIGHT),
            geom=make_geom,
            mass=cfg.TEAPOT_LID_MASS,
            angular_damping=cfg.TEAPOT_LID_ANGULAR_DAMPING
        )
        teapot_lid.create().set_pos_hpr(
            *cls.sample2coords(sample, "teapot_lid")
        )
        teapot_lid.attach_to(scene.graph, scene.world)

        nail = prim.Box(
            "nail",
            cfg.NAIL_LEVER_LWH,
            geom=make_geom,
            mass=cfg.NAIL_LEVER_MASS
        )
        nail.create().set_pos_hpr(
            *cls.sample2coords(sample, "nail")
        )
        nail.attach_to(scene.graph, scene.world)

        nail_pivot = prim.Pivot(
            "pivot",
            nail,
            (-cfg.NAIL_LEVER_LWH[0]*.2, 0, 0),  # magical
            (0, 90, 0),
            geom=make_geom
        )
        nail_pivot.create()
        nail_pivot.attach_to(scene.graph, scene.world)

        top_pulley = prim.RopePulley(
            "top_pulley",
            top_goblet, gate,
            Point3(0), Point3(-cfg.PLANK_LWH[0]/2, 0, 0),
            cfg.TOP_PULLEY_ROPE_LENGTH,
            cls.sample2coords(sample, "top_pulley"),
            geom=make_geom
        )
        top_pulley.create()
        top_pulley.attach_to(scene.graph, scene.world)

        left_pulley = prim.RopePulley(
            "left_pulley",
            left_weight, teapot_lid,
            Point3(-cfg.PLANK_LWH[0]/2, 0, 0),
            Point3(0, 0, cfg.TEAPOT_LID_HEIGHT)/2,
            cfg.LEFT_PULLEY_ROPE_LENGTH,
            cls.sample2coords(sample, "left_pulley"),
            geom=make_geom
        )
        left_pulley.create()
        left_pulley.attach_to(scene.graph, scene.world)

        right_pulley = prim.RopePulleyPivot(
            "right_pulley",
            right_weight, bridge,
            Point3(0, 0, cfg.RIGHT_WEIGHT_HEIGHT/2),
            Point3(0, cfg.PLANK_LWH[1], 0.015),  # magical
            cfg.RIGHT_PULLEY_ROPE_LENGTH,
            cls.sample2coords(sample, "right_pulley"),
            (cfg.RIGHT_PULLEY_PIVOT_HEIGHT, cfg.RIGHT_PULLEY_PIVOT_RADIUS),
            -1, cfg.RIGHT_PULLEY_PIVOT_COILED,
            geom=make_geom
        )
        right_pulley.create()
        right_pulley.attach_to(scene.graph, scene.world)

        return scene

    @staticmethod
    def sample2coords(sample, name):
        if name == "top_track":
            pos = Point3(sample[0], 0, sample[1])
            hpr = Vec3(0, 0, sample[2])
        if name == "ball1":
            at = math.radians(sample[2])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[0] + .005  # magical
                - cfg.TOP_TRACK_LWHT[0]*3/7 * cos_at  # magical
                - cfg.TOP_TRACK_LWHT[2]*3/7 * sin_at,  # magical
                0,
                sample[1]
                + cfg.TOP_TRACK_LWHT[0]*3/7 * sin_at  # magical
                + cfg.TOP_TRACK_LWHT[2]*3/7 * cos_at  # magical
                - (cfg.TOP_TRACK_LWHT[2] - cfg.TOP_TRACK_LWHT[3])
                + cfg.BALL_RADIUS
            )
            hpr = Vec3(0)
        if name == "ball2":
            at = math.radians(sample[2])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[0] + .005  # magical
                - cfg.TOP_TRACK_LWHT[0]/4 * cos_at  # magical
                - cfg.TOP_TRACK_LWHT[2]/4 * sin_at,  # magical
                0,
                sample[1]
                + cfg.TOP_TRACK_LWHT[0]/4 * sin_at  # magical
                + cfg.TOP_TRACK_LWHT[2]/4 * cos_at  # magical
                - (cfg.TOP_TRACK_LWHT[2] - cfg.TOP_TRACK_LWHT[3])
                + cfg.BALL_RADIUS
            )
            hpr = Vec3(0)
        if name == "top_goblet":
            at = math.radians(sample[2])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[0]
                - cfg.TOP_TRACK_LWHT[0]/2 * cos_at
                - cfg.TOP_TRACK_LWHT[2]/2 * sin_at
                + .9*cfg.GOBLET_R1,  # magical
                0,
                sample[1]
                + cfg.TOP_TRACK_LWHT[0]/2 * sin_at
                + cfg.TOP_TRACK_LWHT[2]/2 * cos_at
                + cfg.GOBLET_HEIGHT
            )
            hpr = Vec3(0, 0, 180)
        if name == "middle_track":
            pos = Point3(0)
            hpr = Vec3(0)
        if name == "left_track1":
            at = math.radians(sample[3])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                - cfg.SHORT_TRACK_LWHT[0]/2
                - cfg.LONG_TRACK_LWHT[0]/2 * cos_at,
                0,
                cfg.LONG_TRACK_LWHT[0]/2 * sin_at
            )
            hpr = Vec3(0, 0, sample[3])
        if name == "left_track2":
            pos = Point3(sample[4], 0, sample[5])
            hpr = Vec3(0, 0, sample[6])
        if name == "left_track3":
            pos = Point3(sample[7], 0, sample[8])
            hpr = Vec3(0, 0, sample[9])
        if name == "left_track3_blocker":
            at = math.radians(sample[9])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[7]
                + cfg.SHORT_TRACK_LWHT[0]/2 * cos_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * sin_at
                + cfg.FLAT_SUPPORT_LWH[2],
                0,
                sample[8]
                - cfg.SHORT_TRACK_LWHT[0]/2 * sin_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * cos_at
            )
            hpr = Vec3(0, 0, 90)
        if name == "left_track4":
            pos = Point3(sample[10], 0, sample[11])
            hpr = Vec3(0, 0, sample[12])
        if name == "left_track4_blocker":
            at = math.radians(sample[12])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[10]
                - cfg.LONG_TRACK_LWHT[0]/2 * cos_at
                - cfg.LONG_TRACK_LWHT[2]/2 * sin_at
                - cfg.FLAT_SUPPORT_LWH[2],
                0,
                sample[11]
                + cfg.LONG_TRACK_LWHT[0]/2 * sin_at
                + cfg.LONG_TRACK_LWHT[2]/2 * cos_at
            )
            hpr = Vec3(0, 0, 90)
        if name == "right_track1":
            at = math.radians(sample[13])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                cfg.SHORT_TRACK_LWHT[0]/2
                + cfg.LONG_TRACK_LWHT[0]/2 * cos_at,
                0,
                - cfg.LONG_TRACK_LWHT[0]/2 * sin_at
            )
            hpr = Vec3(0, 0, sample[13])
        if name == "right_track2":
            pos = Point3(sample[14], 0, sample[15])
            hpr = Vec3(0, 0, sample[16])
        if name == "right_track2_blocker1":
            at = math.radians(sample[16])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[14]
                + cfg.SHORT_TRACK_LWHT[0]/2 * cos_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * sin_at
                - cfg.FLAT_SUPPORT_LWH[1]/2,
                cfg.SHORT_TRACK_LWHT[1]/2 + cfg.FLAT_SUPPORT_LWH[2]/2,
                sample[15]
                + cfg.SHORT_TRACK_LWHT[0]/2 * sin_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * cos_at
                + cfg.FLAT_SUPPORT_LWH[0]/2 + cfg.SHORT_TRACK_LWHT[2]
            )
            hpr = Vec3(0, 90, 0)
        if name == "right_track2_blocker2":
            at = math.radians(sample[16])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[14]
                + cfg.SHORT_TRACK_LWHT[0]/2 * cos_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * sin_at
                - cfg.FLAT_SUPPORT_LWH[1]/2,
                - cfg.SHORT_TRACK_LWHT[1]/2 - cfg.FLAT_SUPPORT_LWH[2]/2,
                sample[15]
                + cfg.SHORT_TRACK_LWHT[0]/2 * sin_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * cos_at
                + cfg.FLAT_SUPPORT_LWH[0]/2 + cfg.SHORT_TRACK_LWHT[2]
            )
            hpr = Vec3(0, 90, 0)
        if name == "right_track3":
            pos = Point3(sample[17], 0, sample[18])
            hpr = Vec3(0, 0, sample[19])
        if name == "right_track3_blocker":
            at = math.radians(sample[19])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[17]
                - cfg.SHORT_TRACK_LWHT[0]/2 * cos_at
                - cfg.SHORT_TRACK_LWHT[2]/2 * sin_at
                - cfg.FLAT_SUPPORT_LWH[2],
                0,
                sample[18]
                + cfg.SHORT_TRACK_LWHT[0]/2 * sin_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * cos_at
            )
            hpr = Vec3(0, 0, 90)
        if name == "right_track4":
            pos = Point3(sample[20], 0, sample[21])
            hpr = Vec3(0, 0, sample[22])
        if name == "right_track4_blocker":
            at = math.radians(sample[22])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[20]
                + cfg.SHORT_TRACK_LWHT[0]/2 * cos_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * sin_at
                + cfg.FLAT_SUPPORT_LWH[2],
                0,
                sample[21]
                - cfg.SHORT_TRACK_LWHT[0]/2 * sin_at
                + cfg.SHORT_TRACK_LWHT[2]/2 * cos_at
            )
            hpr = Vec3(0, 0, 90)
        if name == "left_weight":
            pos = Point3(sample[23], 0, sample[24])
            hpr = Vec3(0, 0, 90)
        if name == "left_weight_support":
            pos = Point3(
                sample[23] + cfg.FLAT_SUPPORT_LWH[1]/2 - 0.0021,  # magical
                0,
                sample[24]
                - (cfg.QUAD_PLANK_LWH[0] + cfg.FLAT_SUPPORT_LWH[2]) / 2
            )
            hpr = Vec3(0)
        if name == "right_weight":
            pos = Point3(sample[25], 0, sample[26])
            hpr = Vec3(0)
        if name == "right_weight_support":
            pos = Point3(
                sample[25] - cfg.FLAT_SUPPORT_LWH[0]/2 + 0.001,  # magical
                0,
                sample[26]
                - (cfg.RIGHT_WEIGHT_HEIGHT + cfg.FLAT_SUPPORT_LWH[2]) / 2
            )
            hpr = Vec3(0)
        if name == "gate":
            pos = Point3(sample[27], 0, 0.1605)  # magical
            hpr = Vec3(0, 0, 90)
        if name == "top_weight_support":
            pos = Point3(
                sample[27] + cfg.PLANK_LWH[2]/2
                + cfg.TINY_TRACK_LWH[0]/2 - .001,  # magical
                0,
                0.1605 - cfg.PLANK_LWH[0]/2 - cfg.TINY_TRACK_LWH[2]/2,
            )
            hpr = Vec3(0)
        if name == "top_weight_guide":
            pos = Point3(
                sample[27] - cfg.PLANK_LWH[2]/2
                - cfg.FLAT_SUPPORT_LWH[2]/2 - .003,  # magical
                0,
                0.1605 - cfg.PLANK_LWH[0]/3
            )
            hpr = Vec3(0, 0, 90)
        if name == "sliding_plank":
            pos = Point3(
                sample[27] + cfg.PLANK_LWH[2]/2 + cfg.PLANK_LWH[0]/2,
                0,
                0.1605 - cfg.PLANK_LWH[0]/2 + cfg.PLANK_LWH[1]/2,
            )
            hpr = Vec3(0, 90, 0)
        if name == "nail":
            pos = Point3(  # magical
                sample[27] + cfg.PLANK_LWH[2]/2 + cfg.PLANK_LWH[0] + .0005,
                0,
                0.1605 - cfg.NAIL_LEVER_LWH[0]*.8,  # magical
            )
            hpr = Vec3(0, 0, 90)
        if name == "bridge":
            pos = Point3(sample[28], 0, sample[29])
            hpr = Vec3(0, 0, 90)
        if name == "bottom_goblet":
            pos = Point3(sample[30], 0, sample[31])
            hpr = Vec3(0)
        if name == "teapot_base":
            pos = Point3(sample[32], 0, sample[31])
            hpr = Vec3(0)
        if name == "teapot_lid":
            pos = Point3(
                sample[32],
                0,
                sample[31] + cfg.GOBLET_HEIGHT + cfg.TEAPOT_LID_HEIGHT/2
            )
            hpr = Vec3(0)
        if name == "top_pulley":
            at = math.radians(sample[2])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos1 = Point3(
                sample[0]
                - cfg.TOP_TRACK_LWHT[0]/2 * cos_at
                - cfg.TOP_TRACK_LWHT[2]/2 * sin_at
                + .9*cfg.GOBLET_R1,  # magical
                0,
                sample[33]
            )
            pos2 = Point3(sample[27], 0, sample[33])
            return pos1, pos2
        if name == "left_pulley":
            pos1 = Point3(sample[23], 0, sample[34]+.03)  # magical
            pos2 = Point3(sample[32], 0, sample[34])
            return pos1, pos2
        if name == "right_pulley":
            pos1 = Point3(sample[25], 0, sample[35])
            pos2 = Point3(sample[36], 0, sample[29])
            return pos1, pos2
        return pos, hpr

    # LEGACY
    @classmethod
    def check_valid(cls, scene, world):
        return cls._check_physically_valid_scene(Scene(scene, world))

    init_scenario = init_scene
