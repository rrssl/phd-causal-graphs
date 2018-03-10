"""
Main

"""
import math
import os
import sys

import numpy as np
from panda3d.core import Point2, Point3, Vec2, Vec3
import scipy.optimize as opt
from sklearn.externals.joblib import delayed, Parallel

sys.path.insert(0, os.path.abspath("../.."))
import xp.config as cfg  # noqa: E402
from core.primitives import Ball, Box, DominoRun, Lever  # noqa: E402
from xp.dominoes.geom import tilt_box_forward  # noqa: E402
from xp.dominoes.templates import create_branch, create_line  # noqa: E402
from xp.simulate import Simulation  # noqa: E402

from xp.scenarios import (AndTerminationCondition,  # noqa: E402
                          DominoRunTerminationCondition,
                          DominoRunTopplingTimeObserver,
                          init_scene)


OPTIM_SIMU_TIMESTEP = 1e-3
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


class BallRunSubmodel:
    min_z = 0
    max_z = LEVER_HEIGHT
    min_r = 0
    max_r = 90

    @classmethod
    def opt2model(cls, x_opt):
        """Map variables from [0, 1] to their original interval."""
        z = x_opt[0]*(cls.max_z - cls.min_z) + cls.min_z
        r = x_opt[1]*(cls.max_r - cls.min_r) + cls.min_r
        return z, r

    @classmethod
    def model2opt(cls, x_model):
        """Map variables from their original interval to [0, 1]."""
        z, r = x_model
        x = np.array([
            (z - cls.min_z) / (cls.max_z - cls.min_z),
            (r - cls.min_r) / (cls.max_r - cls.min_r),
        ])
        return x

    @classmethod
    def update(cls, scene, x):
        z, r = cls.opt2model(x)
        # Find relevant objects in the scene.
        ball = scene.find("ball*")
        preplank = scene.find("preplank*")
        plank = scene.find("plank*")
        # Update their coordinates.
        ball.set_z(z)
        preplank.set_z(ball.get_z() - cfg.BALL_RADIUS - PREPLANK_EXTENTS[2]/2)
        plank.set_r(r)
        r_rad = math.radians(r)
        plank.set_x(
            PREPLANK_POS.x + PREPLANK_EXTENTS[0]/2
            + cfg.PLANK_LENGTH/2*math.cos(r_rad)
            - cfg.PLANK_THICKNESS/2*math.sin(r_rad)
        )
        plank.set_z(preplank.get_z() - cfg.PLANK_LENGTH/2*math.sin(r_rad))
        # Update Bullet nodes' transforms.
        for obj in (ball, preplank, plank):
            obj.node().set_transform_dirty()


class Model:
    """Interface between the optimizer and the objects being optimized."""
    def __init__(self):
        self.scenario = None
        self.left_time = None
        self.right_time = None

    @staticmethod
    def opt2model(x_opt):
        """Map variables from [0, 1] to their original interval."""
        return BallRunSubmodel.opt2model(x_opt)

    @staticmethod
    def model2opt(x_model):
        """Map variables from their original interval to [0, 1]."""
        return BallRunSubmodel.model2opt(x_model)

    def update(self, x, _visual=False):
        if not np.isfinite(x).all():
            self.left_time = math.nan
            self.right_time = math.nan
            return
        if _visual:
            scenario = DominoesBallSync(make_geom=True)
            BallRunSubmodel.update(scenario.scene, x)
            simu = Simulation(scenario, timestep=OPTIM_SIMU_TIMESTEP)
            simu.run_visual()
        # Create new scenario.
        scenario = DominoesBallSync(make_geom=_visual)
        self.scenario = scenario
        scene = scenario.scene
        # Update with new parameters.
        BallRunSubmodel.update(scene, x)
        # Create observers.
        observers = (
            DominoRunTopplingTimeObserver(scene.find("left_row*")),
            DominoRunTopplingTimeObserver(scene.find("right_row*"))
        )
        # Create and run simulation.
        simu = Simulation(scenario, observers, timestep=OPTIM_SIMU_TIMESTEP)
        simu.run()
        print("Observed times -- left:", observers[0].times)
        print("Observed times -- right:", observers[1].times)
        # Update output variables.
        self.left_time = observers[0].times[-1]
        self.right_time = observers[1].times[-1]


class Objective:
    err_val = 10

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        self.model.update(x, _visual=0)
        f = (self.model.left_time - self.model.right_time) ** 2
        if not np.isfinite(f):
            f = self.err_val
        print("Internal call to objective -- x = {}, f = {}".format(x, f))
        return f


class NonPenetrationConstraint:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        # self.model.update(x)
        return 0


def main():
    model = Model()
    objective = Objective(model)
    x_init = model.model2opt([BALL_POS.z, PLANK_HPR.z])
    bounds = [(.0, 1)] * len(x_init)
    # constraints = (
    #     {'type': 'ineq', 'fun': NonPenetrationConstraint(model)},
    # )
    # First, brute force.
    ns = 51
    x_grid = np.mgrid[[np.s_[b[0]:b[1]:ns*1j] for b in bounds]]
    x_grid = x_grid.reshape(2, -1).T
    filename = "grid_values.npy"
    try:
        vals = np.load(filename)
    except FileNotFoundError:
        vals = Parallel(n_jobs=6)(delayed(objective)(x) for x in x_grid)
        np.save(filename, vals)
    x_best = x_grid[np.argmin(vals)]
    vals = np.asarray(vals).reshape(ns, ns)

    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    im = plt.imshow(
        vals.T, origin='lower',
        extent=np.concatenate(bounds), aspect='equal'
    )
    plt.colorbar(im)
    plt.scatter(*x_best)
    plt.show()

    x_init = x_best
    x_best = opt.minimize(objective, x_init,
                          bounds=bounds,
                          # constraints=constraints,
                          options=dict(disp=True, maxiter=10, eps=1e-4)).x
    print(x_best)
    print(objective(x_best))

    # Show the solution.
    model.update(x_best, _visual=True)


if __name__ == "__main__":
    main()
