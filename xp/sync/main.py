"""
Main

"""
import os
import sys

from panda3d.core import NodePath

sys.path.insert(0, os.path.abspath("../.."))
import xp.config as cfg  # noqa: E402
from core.primitives import Ball, Box, DominoRun, Lever, Plane  # noqa: E402
from gui.viewers import PhysicsViewer  # noqa: E402
from xp.dominoes.templates import create_branch, create_line  # noqa: E402


def main():
    app = PhysicsViewer()
    world = app.world
    scene = NodePath("scene")

    floor = Plane("floor", geom=False)
    floor.create()
    floor.attach_to(scene, world)

    branch = DominoRun(
        "branch",
        cfg.DOMINO_EXTENTS,
        create_branch(),
        geom=True,
        mass=cfg.DOMINO_MASS
    )
    branch.create()
    branch.attach_to(scene, world)

    left_row = DominoRun(
        "left_row",
        cfg.DOMINO_EXTENTS,
        create_line(),
        geom=True,
        mass=cfg.DOMINO_MASS,
    )
    left_row.create()
    left_row.attach_to(scene, world)

    lever = Lever(
        "lever",
        (),
        geom=True,
        mass=cfg.PLANK_MASS
    )
    lever.create()
    lever.attach_to(scene, world)

    ball = Ball(
        "ball",
        cfg.BALL_RADIUS,
        None,
        geom=True,
        mass=cfg.BALL_MASS
    )
    ball.create()
    ball.attach_to(scene, world)

    plank = Box(
        "plank",
        cfg.PLANK,
        None,
        geom=True,
        mass=cfg.PLANK_MASS
    )
    plank.create()
    plank.attach_to(scene, world)

    right_row = DominoRun(
        "right_row",
        cfg.DOMINO_EXTENTS,
        create_line(),
        geom=True,
        mass=cfg.DOMINO_MASS,
    )
    right_row.create()
    right_row.attach_to(scene, world)

    app.run()


if __name__ == "__main__":
    main()
