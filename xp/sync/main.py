"""
Main

"""
import math
import os
import sys

from panda3d.core import Point2, Point3, Vec2, Vec3

sys.path.insert(0, os.path.abspath("../.."))
import xp.config as cfg  # noqa: E402
from core.primitives import Ball, Box, DominoRun, Lever  # noqa: E402
from gui.viewers import ScenarioViewer  # noqa: E402
from xp.dominoes.geom import tilt_box_forward  # noqa: E402
from xp.dominoes.templates import create_branch, create_line  # noqa: E402
from xp.scenarios import (AndTerminationCondition,  # noqa: E402
                          DominoRunTerminationCondition, init_scene)


# TODO. Turn this into a list of dicts.
BRANCH_ORIGIN = Point2(0)
BRANCH_ANGLE = 0
BRANCH_HALF_LENGTH = .1
BRANCH_HALF_WIDTH = .07
BRANCH_DENSITY = 6 / BRANCH_HALF_LENGTH
LEFT_ROW_ORIGIN = BRANCH_ORIGIN + Vec2(2*BRANCH_HALF_LENGTH + 3 * cfg.t,
                                       BRANCH_HALF_WIDTH)
LEFT_ROW_ANGLE = BRANCH_ANGLE
LEFT_ROW_LENGTH = .2
LEFT_ROW_DENSITY = 12 / LEFT_ROW_LENGTH
LEVER_THICKNESS = .005
LEVER_WIDTH = .02
LEVER_HEIGHT = .1
LEVER_EXTENTS = LEVER_THICKNESS, LEVER_WIDTH, LEVER_HEIGHT
LEVER_MASS = .01
LEVER_POS = Point3(LEFT_ROW_ORIGIN.x, -BRANCH_HALF_WIDTH, LEVER_HEIGHT / 2)
LEVER_PIVOT_POS_HPR = (LEVER_THICKNESS/2, 0, -LEVER_HEIGHT/2, 0, 90, 0)
BALL_POS = Point3(LEVER_POS.x + cfg.BALL_RADIUS + .01, -BRANCH_HALF_WIDTH, .09)
PREPLANK_EXTENTS = .01, cfg.PLANK_WIDTH, cfg.PLANK_THICKNESS
PREPLANK_POS = Point3(BALL_POS.x + PREPLANK_EXTENTS[0]/2 - .005,
                      BALL_POS.y,
                      BALL_POS.z - cfg.BALL_RADIUS - PREPLANK_EXTENTS[2]/2)
PREPLANK_HPR = Vec3(0)
PLANK_HPR = Vec3(0, 0, 25)
PLANK_POS = Point3(PREPLANK_POS.x + PREPLANK_EXTENTS[0]/2
                   + cfg.PLANK_LENGTH/2*math.cos(math.radians(PLANK_HPR.z))
                   - cfg.PLANK_THICKNESS/2*math.sin(math.radians(PLANK_HPR.z)),
                   BALL_POS.y,
                   PREPLANK_POS.z
                   - cfg.PLANK_LENGTH/2*math.sin(math.radians(PLANK_HPR.z)))
RIGHT_ROW_LENGTH = .05
RIGHT_ROW_ORIGIN = Point2(
    LEFT_ROW_ORIGIN.x + LEFT_ROW_LENGTH - RIGHT_ROW_LENGTH, -BRANCH_HALF_WIDTH)
RIGHT_ROW_ANGLE = BRANCH_ANGLE
RIGHT_ROW_DENSITY = 4 / RIGHT_ROW_LENGTH


class DominoesBallSync:
    """Scenario used to sync a domino run with a ball run."""
    def __init__(self, make_geom=False, **kwargs):
        self.scene, self.world = self.init_scenario(make_geom)
        term1 = DominoRunTerminationCondition(self.scene.find("left_row"))
        term2 = DominoRunTerminationCondition(self.scene.find("right_row"))
        self.terminate = AndTerminationCondition((term1, term2))

    @staticmethod
    def init_scenario(make_geom=False):
        scene, world = init_scene()

        branch = DominoRun(
            "branch",
            cfg.DOMINO_EXTENTS,
            create_branch(BRANCH_ORIGIN, BRANCH_ANGLE, BRANCH_HALF_LENGTH,
                          BRANCH_HALF_WIDTH, BRANCH_DENSITY),
            geom=make_geom,
            mass=cfg.DOMINO_MASS
        )
        branch.create()
        tilt_box_forward(branch.path.get_child(0), cfg.TOPPLING_ANGLE+1)
        branch.attach_to(scene, world)

        left_row = DominoRun(
            "left_row",
            cfg.DOMINO_EXTENTS,
            create_line(LEFT_ROW_ORIGIN, LEFT_ROW_ANGLE, LEFT_ROW_LENGTH,
                        LEFT_ROW_DENSITY),
            geom=make_geom,
            mass=cfg.DOMINO_MASS,
        )
        left_row.create()
        left_row.attach_to(scene, world)

        lever = Lever(
            "lever",
            LEVER_EXTENTS,
            LEVER_PIVOT_POS_HPR,
            geom=make_geom,
            mass=LEVER_MASS
        )
        lever.create().set_pos(LEVER_POS)
        lever.attach_to(scene, world)

        ball = Ball(
            "ball",
            cfg.BALL_RADIUS,
            geom=make_geom,
            mass=cfg.BALL_MASS
        )
        ball.create().set_pos(BALL_POS)
        ball.attach_to(scene, world)

        preplank = Box(
            "preplank",
            PREPLANK_EXTENTS,
            geom=make_geom
        )
        preplank.create().set_pos_hpr(PREPLANK_POS, PREPLANK_HPR)
        preplank.attach_to(scene, world)

        plank = Box(
            "plank",
            cfg.PLANK_EXTENTS,
            geom=make_geom
        )
        plank.create().set_pos_hpr(PLANK_POS, PLANK_HPR)
        plank.attach_to(scene, world)

        right_row = DominoRun(
            "right_row",
            cfg.DOMINO_EXTENTS,
            create_line(RIGHT_ROW_ORIGIN, RIGHT_ROW_ANGLE, RIGHT_ROW_LENGTH,
                        RIGHT_ROW_DENSITY),
            geom=make_geom,
            mass=cfg.DOMINO_MASS,
        )
        right_row.create()
        right_row.attach_to(scene, world)

        return scene, world

    def succeeded(self):
        return self.terminate.status == 'success'


def main():
    scenario = DominoesBallSync(make_geom=True)
    app = ScenarioViewer(scenario, frame_rate=1000)
    app.run()


if __name__ == "__main__":
    main()
