from collections import namedtuple
import math

import numpy as np
from panda3d.core import NodePath, Point3, Vec3

import core.primitives as prim
import xp.ifelse.config as cfg
from xp.scenarios import DummyTerminationCondition, Scenario
from core.export import VectorFile


Scene = namedtuple('Scene', ['graph', 'world'])


def init_scene():
    """Initialize the Panda3D scene."""
    graph = NodePath("scene")
    world = prim.World()
    world.set_gravity(cfg.GRAVITY)
    return Scene(graph, world)


class ConditionalBallRun(Scenario):
    """Scenario

    Parameters
    ----------
    sample : (8,) sequence
      [xt, zt, at, zb, ab, xg, zg, ag]

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        self._scene = self.init_scene(sample, make_geom)
        self.causal_graph = self.init_causal_graph(self._scene)
        # LEGACY
        self.world = self._scene.world
        self.scene = self._scene.graph
        self.terminate = self.causal_graph

    @staticmethod
    def check_valid(scene):
        return True

    @staticmethod
    def get_distribution():
        return None

    @staticmethod
    def get_robustness(sample, alpha):
        return 0

    @staticmethod
    def export_scenario(filename, sample, sheetsize):
        coords = np.array([])
        sizes = np.array([])

        xy = coords[:, :2] * 100
        xy = xy - (xy.min(axis=0) + xy.max(axis=0))/2 + np.asarray(sheetsize)/2
        a = coords[:, 2]
        sizes *= 100

        vec = VectorFile(filename, sheetsize)
        vec.add_rectangles(xy, a, sizes)
        vec.save()

    @staticmethod
    def init_causal_graph(scene):
        return DummyTerminationCondition()

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
