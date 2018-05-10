from collections import namedtuple
import math
import os
import pickle
import subprocess

import chaospy as cp
import numpy as np
from panda3d.core import NodePath, Point3, Vec3

import core.primitives as prim
import xp.causal as causal
import xp.ifelse.config as cfg
from xp.scenarios import CausalGraphTerminationCondition, Samplable, Scenario
from core.export import VectorFile


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
        linvel = self.body.node().get_linear_velocity().length_squared()
        angvel = self.body.node().get_angular_velocity().length_squared()
        return (linvel < cfg.STOPPING_LINEAR_VELOCITY
                and angvel < cfg.STOPPING_ANGULAR_VELOCITY)


class StateObserver:
    """Keeps track of the full state of each non-static object in the scene."""
    def __init__(self, scene):
        self.graph_root = scene.graph
        self.paths = []
        self.states = dict()
        # Find and tag any GeomNode child of a non-static BulletRigidBodyNode.
        for body in scene.world.get_rigid_bodies():
            if not body.is_static():
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
            self.states[anim_id].append([time, x, y, z, w, i, j, k])

    def export(self, filename):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.states, f)


class ConditionalBallRun(Samplable, Scenario):
    """Scenario

    Parameters
    ----------
    sample : (8,) sequence
      [xt, zt, at, zb, ab, xg, zg, ag]

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        self._scene = self.init_scene(sample, make_geom)
        self.causal_graph = self.init_causal_graph(self._scene,
                                                   verbose=make_geom)
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
        graph = scene.graph
        top_track = graph.find("top_track*").node()
        bottom_track = graph.find("bottom_track*").node()
        high_plank = graph.find("high_plank*").node()
        low_plank = graph.find("low_plank*").node()
        base_plank = graph.find("base_plank*").node()
        goblet = graph.find("goblet*").node()
        # Enable collisions for static objects
        for body in (top_track, bottom_track, goblet):
            body.set_static(False)
            body.set_active(True)
        test_pairs = [
            (top_track, bottom_track),
            (top_track, high_plank),
            (top_track, low_plank),
            (top_track, goblet),
            (bottom_track, base_plank),
            (bottom_track, goblet),
            (low_plank, goblet),
            (base_plank, goblet)
        ]
        world = scene.world
        return not any(world.contact_test_pair(a, b).get_num_contacts()
                       for a, b in test_pairs)

    @staticmethod
    def get_distribution():
        distributions = [
            cp.Truncnorm(xmin, xmax, (xmin+xmax)/2, (xmax-xmin)/4)
            for xmin, xmax in cfg.SCENARIO_PARAMETERS_BOUNDS
        ]
        return cp.J(*distributions)

    def export_scene_to_egg(self, filename):
        if filename[-4:] == ".egg":
            filename = filename[:-4]
        self._scene.graph.write_bam_file(filename + ".bam")
        subprocess.run(["bam2egg", "-o", filename + ".egg", filename + ".bam"])
        os.remove(filename + ".bam")

    @classmethod
    def export_scene_to_pdf(cls, filename, sample, sheetsize):
        coords = []
        sizes = []
        shapes = []
        pos, hpr = cls.sample2coords(sample, "top_track")
        coords.append([pos.x, pos.z, hpr.z])
        sizes.append([cfg.TOP_TRACK_LWH[0], cfg.TOP_TRACK_LWH[2]])
        shapes.append('rect')
        pos, hpr = cls.sample2coords(sample, "bottom_track")
        coords.append([pos.x, pos.z, hpr.z])
        sizes.append([cfg.BOTTOM_TRACK_LWH[0], cfg.BOTTOM_TRACK_LWH[2]])
        shapes.append('rect')
        pos, hpr = cls.sample2coords(sample, "high_plank")
        coords.append([pos.x, pos.z, hpr.z])
        sizes.append([cfg.HIGH_PLANK_LWH[0], cfg.HIGH_PLANK_LWH[2]])
        shapes.append('rect')
        pos, hpr = cls.sample2coords(sample, "low_plank")
        coords.append([pos.x, pos.z, hpr.z])
        sizes.append([cfg.LOW_PLANK_LWH[0], cfg.LOW_PLANK_LWH[2]])
        shapes.append('rect')
        pos, hpr = cls.sample2coords(sample, "base_plank")
        coords.append([pos.x, pos.z, hpr.z])
        sizes.append([cfg.BASE_PLANK_LWH[0], cfg.BASE_PLANK_LWH[2]])
        shapes.append('rect')
        pos, hpr = cls.sample2coords(sample, "flat_support")
        coords.append([pos.x, pos.z, hpr.z])
        sizes.append([cfg.FLAT_SUPPORT_LWH[0], cfg.FLAT_SUPPORT_LWH[2]])
        shapes.append('rect')
        pos, _ = cls.sample2coords(sample, "round_support")
        coords.append([pos.x, pos.z, 0])
        sizes.append([cfg.ROUND_SUPPORT_RADIUS, 0])
        shapes.append('circ')
        pos, hpr = cls.sample2coords(sample, "goblet")
        angle = math.radians(hpr.z)
        coords.append([pos.x + math.sin(angle)*cfg.GOBLET_HEIGHT/2,
                       pos.z + math.cos(angle)*cfg.GOBLET_HEIGHT/2,
                       hpr.z])
        sizes.append([cfg.GOBLET_R1*2, cfg.GOBLET_HEIGHT])
        shapes.append('rect')
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
        ball = scene_graph.find("ball*")
        top_track = scene_graph.find("top_track*")
        bottom_track = scene_graph.find("bottom_track*")
        high_plank = scene_graph.find("high_plank*")
        low_plank = scene_graph.find("low_plank*")
        base_plank = scene_graph.find("base_plank*")
        goblet = scene_graph.find("goblet*")

        ball_rolls_on_top_track = causal.Event(
            "ball_rolls_on_top_track",
            RollingOn(ball, top_track, world),
            None,
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball_hits_high_plank = causal.Event(
            "ball_hits_high_plank",
            Contact(ball, high_plank, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        high_plank_topples = causal.Event(
            "high_plank_topples",
            Toppling(high_plank, cfg.HIGH_PLANK_TOPPLING_ANGLE),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball_rolls_on_bottom_track = causal.Event(
            "ball_rolls_on_bottom_track",
            RollingOn(ball, bottom_track, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        base_plank_moves = causal.Event(
            "base_plank_moves",
            Pivoting(base_plank),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        low_plank_falls = causal.Event(
            "low_plank_falls",
            NoContact(low_plank, base_plank, world),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball_enters_goblet = causal.Event(
            "ball_enters_goblet",
            Inclusion(ball, goblet),
            causal.AllBefore(),
            causal.AllAfter(verbose=verbose),
            verbose=verbose
        )
        ball_stops = causal.Event(
            "ball_stops",
            Stopping(ball),
            causal.AllBefore(),
            None,
            verbose=verbose
        )
        causal.connect(ball_rolls_on_top_track, ball_hits_high_plank),
        causal.connect(ball_hits_high_plank, high_plank_topples),
        causal.connect(ball_hits_high_plank, ball_rolls_on_bottom_track),
        causal.connect(ball_rolls_on_bottom_track, ball_enters_goblet),
        causal.connect(high_plank_topples, base_plank_moves),
        causal.connect(base_plank_moves, low_plank_falls),
        causal.connect(low_plank_falls, ball_enters_goblet)
        causal.connect(ball_enters_goblet, ball_stops)

        graph = causal.CausalGraphTraverser(
            root=ball_rolls_on_top_track, verbose=verbose
        )
        return graph

    @classmethod
    def init_scene(cls, sample, make_geom=False):
        scene = init_scene()

        ball = prim.Ball(
            "ball",
            cfg.BALL_RADIUS,
            geom=make_geom,
            mass=cfg.BALL_MASS
        )
        ball.create().set_pos_hpr(
            *cls.sample2coords(sample, "ball")
        )
        ball.attach_to(scene.graph, scene.world)

        top_track = prim.Box(
            "top_track",
            cfg.TOP_TRACK_LWH,
            geom=make_geom
        )
        top_track.create().set_pos_hpr(
            *cls.sample2coords(sample, "top_track")
        )
        top_track.attach_to(scene.graph, scene.world)

        bottom_track = prim.Box(
            "bottom_track",
            cfg.BOTTOM_TRACK_LWH,
            geom=make_geom
        )
        bottom_track.create().set_pos_hpr(
            *cls.sample2coords(sample, "bottom_track")
        )
        bottom_track.attach_to(scene.graph, scene.world)

        high_plank = prim.Box(
            "high_plank",
            cfg.HIGH_PLANK_LWH,
            geom=make_geom,
            mass=cfg.HIGH_PLANK_MASS
        )
        high_plank.create().set_pos_hpr(
            *cls.sample2coords(sample, "high_plank")
        )
        high_plank.attach_to(scene.graph, scene.world)

        low_plank = prim.Box(
            "low_plank",
            cfg.LOW_PLANK_LWH,
            geom=make_geom,
            mass=cfg.LOW_PLANK_MASS
        )
        low_plank.create().set_pos_hpr(
            *cls.sample2coords(sample, "low_plank")
            )
        low_plank.attach_to(scene.graph, scene.world)

        base_plank = prim.Box(
            "base_plank",
            cfg.BASE_PLANK_LWH,
            geom=make_geom,
            mass=cfg.BASE_PLANK_MASS
        )
        base_plank.create().set_pos_hpr(
            *cls.sample2coords(sample, "base_plank")
        )
        base_plank.attach_to(scene.graph, scene.world)

        flat_support = prim.Box(
            "flat_support",
            cfg.FLAT_SUPPORT_LWH,
            geom=make_geom
        )
        flat_support.create().set_pos_hpr(
            *cls.sample2coords(sample, "flat_support")
        )
        flat_support.attach_to(scene.graph, scene.world)

        round_support = prim.Cylinder(
            "round_support",
            (cfg.ROUND_SUPPORT_RADIUS, cfg.ROUND_SUPPORT_HEIGHT),
            geom=make_geom
        )
        round_support.create().set_pos_hpr(
            *cls.sample2coords(sample, "round_support")
        )
        round_support.attach_to(scene.graph, scene.world)

        goblet = prim.Goblet(
            "goblet",
            (cfg.GOBLET_HEIGHT, cfg.GOBLET_R1, cfg.GOBLET_R2),
            geom=make_geom
        )
        goblet.create().set_pos_hpr(
            *cls.sample2coords(sample, "goblet")
        )
        goblet.attach_to(scene.graph, scene.world)

        return scene

    @staticmethod
    def sample2coords(sample, name):
        if name == "ball":
            at = math.radians(sample[2])
            sin_at = math.sin(at)
            cos_at = math.cos(at)
            pos = Point3(
                sample[0] - .001
                + cfg.TOP_TRACK_LWH[0]/2 * cos_at
                + cfg.TOP_TRACK_LWH[2]/2 * sin_at,
                0,
                sample[1]
                - cfg.TOP_TRACK_LWH[0]/2 * sin_at
                + cfg.TOP_TRACK_LWH[2]/2 * cos_at
                + cfg.BALL_RADIUS
            )
            hpr = Vec3(0)
        if name == "top_track":
            pos = Point3(sample[0], 0, sample[1])
            hpr = Vec3(0, 0, sample[2])
        if name == "bottom_track":
            pos = Point3(0, 0, sample[3])
            hpr = Vec3(0, 0, sample[4])
        if name == "high_plank":
            ab = math.radians(sample[4])
            sin_ab = math.sin(ab)
            cos_ab = math.cos(ab)
            pos = Point3(
                -cfg.HIGH_PLANK_LWH[2]/2
                - cfg.BOTTOM_TRACK_LWH[0]/2 * cos_ab
                - cfg.BOTTOM_TRACK_LWH[2]/2 * sin_ab,
                0,
                cfg.ROUND_SUPPORT_RADIUS + cfg.BASE_PLANK_LWH[2]
                + cfg.HIGH_PLANK_LWH[0]/2
            )
            hpr = Vec3(0, 0, 90)
        if name == "low_plank":
            ab = math.radians(sample[4])
            sin_ab = math.sin(ab)
            cos_ab = math.cos(ab)
            pos = Point3(
                cfg.LOW_PLANK_LWH[2]/2
                + cfg.BOTTOM_TRACK_LWH[0]/2 * cos_ab
                + cfg.BOTTOM_TRACK_LWH[2]/2 * sin_ab,
                0,
                cfg.ROUND_SUPPORT_RADIUS + cfg.BASE_PLANK_LWH[2]
                + cfg.LOW_PLANK_LWH[0]/2
            )
            hpr = Vec3(0, 0, 90)
        if name == "base_plank":
            pos = Point3(
                0,
                0,
                cfg.ROUND_SUPPORT_RADIUS + cfg.BASE_PLANK_LWH[2]/2
            )
            hpr = Vec3(0)
        if name == "flat_support":
            pos = Point3(
                -cfg.BASE_PLANK_LWH[0]/2 + cfg.FLAT_SUPPORT_LWH[0]/2,
                0,
                cfg.ROUND_SUPPORT_RADIUS - cfg.FLAT_SUPPORT_LWH[2]/2
            )
            hpr = Vec3(0)
        if name == "round_support":
            pos = Point3(0)
            hpr = Vec3(0, 90, 0)
        if name == "goblet":
            pos = Point3(sample[5], 0, sample[6])
            hpr = Vec3(0, 0, sample[7])
        return pos, hpr

    # LEGACY
    @classmethod
    def check_valid(cls, scene, world):
        return cls._check_physically_valid_scene(Scene(scene, world))

    init_scenario = init_scene
