"""
Main

"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from joblib import delayed, Parallel
from panda3d.core import Point2, Point3, Vec2, Vec3
from matplotlib.cm import get_cmap
get_cmap().set_bad(color='red')

sys.path.insert(0, os.path.abspath("../.."))
import xp.config as cfg  # noqa: E402
from core.export import VectorFile  # noqa: E402
from core.primitives import Ball, Box, DominoRun, Lever  # noqa: E402
from xp.dominoes.geom import tilt_box_forward  # noqa: E402
from xp.dominoes.templates import (create_branch, create_line,  # noqa: E402
                                   create_wave, create_x_switch)
from xp.domino_predictors import DominoRobustness2  # noqa: E402
from xp.simulate import Simulation  # noqa: E402

from xp.scenarios import (AndTerminationCondition,  # noqa: E402
                          DominoRunTerminationCondition,
                          DominoRunTopplingTimeObserver,
                          MoveCollideTerminationCondition,
                          init_scene)


OPTIM_SIMU_TIMESTEP = 1/2000
# TODO. Turn this into a list of dicts.
BRANCH_ORIGIN = Point2(0)
BRANCH_ANGLE = 0
BRANCH_LENGTH = .2
BRANCH_WIDTH = .14
BRANCH_NDOMS = 6
LEFT_ROW_ORIGIN = BRANCH_ORIGIN + Vec2(BRANCH_LENGTH + 3 * cfg.t,
                                       BRANCH_WIDTH / 2)
LEFT_ROW_ANGLE = BRANCH_ANGLE
LEFT_ROW_LENGTH = .45
LEFT_ROW_WIDTH = .06
LEFT_ROW_NDOMS = 17
LEVER_THICKNESS = .005
LEVER_WIDTH = .02
LEVER_HEIGHT = .29
LEVER_EXTENTS = LEVER_THICKNESS, LEVER_WIDTH, LEVER_HEIGHT
LEVER_MASS = .005
LEVER_ANGULAR_DAMPING = .3
LEVER_POS = Point3(LEFT_ROW_ORIGIN.x, -BRANCH_WIDTH / 2, LEVER_HEIGHT / 2)
LEVER_PIVOT_POS_HPR = (LEVER_THICKNESS/2, 0, -LEVER_HEIGHT/2, 0, 90, 0)
BALL_POS = Point3(LEVER_POS.x + LEVER_THICKNESS/2 + cfg.BALL_RADIUS + .001,
                  -BRANCH_WIDTH / 2, .2)
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
RIGHT_ROW_LENGTH = .1
RIGHT_ROW_ORIGIN = Point2(
    LEFT_ROW_ORIGIN.x + LEFT_ROW_LENGTH - RIGHT_ROW_LENGTH, -BRANCH_WIDTH / 2)
RIGHT_ROW_ANGLE = BRANCH_ANGLE
RIGHT_ROW_NDOMS = 5
SWITCH_ORIGIN = LEFT_ROW_ORIGIN + Vec2(LEFT_ROW_LENGTH + .03, 0)
SWITCH_ANGLE = 0


class DominoesBallSync:
    """Scenario used to sync a domino run with a ball run.

    Parameters
    ----------
    sample : (5,) sequence
      [ndoms*, wd, bx, bz, pa] (*: integers)

    """
    def __init__(self, sample, make_geom=False, verbose_cond=False, **kwargs):
        self.scene, self.world = self.init_scenario(sample, make_geom)
        term1 = DominoRunTerminationCondition(
            self.scene.find("left_row"), verbose=verbose_cond
        )
        term2 = MoveCollideTerminationCondition(
            self.world,
            self.scene.find("lever*").get_child(0),
            self.scene.find("ball*"),
            verbose=verbose_cond
        )
        term3 = DominoRunTerminationCondition(
            self.scene.find("right_row"), verbose=verbose_cond
        )
        self.terminate = AndTerminationCondition((term1, term2, term3),
                                                 verbose=verbose_cond)

    @staticmethod
    def init_scenario(sample, make_geom=False):
        scene, world = init_scene()

        branch = DominoRun(
            "branch",
            cfg.DOMINO_EXTENTS,
            create_branch(BRANCH_ORIGIN, BRANCH_ANGLE, BRANCH_LENGTH,
                          BRANCH_WIDTH, BRANCH_NDOMS),
            geom=make_geom,
            mass=cfg.DOMINO_MASS
        )
        branch.create()
        tilt_box_forward(branch.path.get_child(0), cfg.TOPPLING_ANGLE+1)
        branch.attach_to(scene, world)

        coords = create_wave(LEFT_ROW_ORIGIN, LEFT_ROW_ANGLE, LEFT_ROW_LENGTH,
                             sample[1], int(sample[0]))
        left_row = DominoRun(
            "left_row",
            cfg.DOMINO_EXTENTS,
            coords,
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
            mass=LEVER_MASS,
            angular_damping=LEVER_ANGULAR_DAMPING
        )
        lever.create().set_pos(LEVER_POS)
        lever.attach_to(scene, world)

        ball = Ball(
            "ball",
            cfg.BALL_RADIUS,
            geom=make_geom,
            mass=cfg.BALL_MASS
        )
        ball_pos = Point3(sample[2], BALL_POS.y, sample[3])
        ball.create().set_pos(ball_pos)
        ball.attach_to(scene, world)

        preplank = Box(
            "preplank",
            PREPLANK_EXTENTS,
            geom=make_geom
        )
        preplank_pos = Point3(
            ball_pos.x + PREPLANK_EXTENTS[0]/2 - .005,
            PREPLANK_POS.y,
            ball_pos.z - cfg.BALL_RADIUS - PREPLANK_EXTENTS[2]/2
        )
        preplank.create().set_pos_hpr(preplank_pos, PREPLANK_HPR)
        preplank.attach_to(scene, world)

        plank = Box(
            "plank",
            cfg.PLANK_EXTENTS,
            geom=make_geom
        )
        r_rad = math.radians(sample[4])
        plank_pos = Point3(
            preplank_pos.x + PREPLANK_EXTENTS[0]/2
            + cfg.PLANK_LENGTH/2*math.cos(r_rad)
            - cfg.PLANK_THICKNESS/2*math.sin(r_rad),
            PLANK_POS.y,
            preplank_pos.z - cfg.PLANK_LENGTH/2*math.sin(r_rad)
        )
        plank_hpr = Vec3(PLANK_HPR.x, PLANK_HPR.y, sample[4])
        plank.create().set_pos_hpr(plank_pos, plank_hpr)
        plank.attach_to(scene, world)

        right_row = DominoRun(
            "right_row",
            cfg.DOMINO_EXTENTS,
            create_line(RIGHT_ROW_ORIGIN, RIGHT_ROW_ANGLE, RIGHT_ROW_LENGTH,
                        RIGHT_ROW_NDOMS),
            geom=make_geom,
            mass=cfg.DOMINO_MASS,
        )
        right_row.create()
        right_row.attach_to(scene, world)

        switch = DominoRun(
            "switch",
            cfg.DOMINO_EXTENTS,
            create_x_switch(SWITCH_ORIGIN, SWITCH_ANGLE, BRANCH_WIDTH,
                            11),
            geom=make_geom,
            mass=cfg.DOMINO_MASS,
        )
        switch.create()
        switch.attach_to(scene, world)

        return scene, world

    @staticmethod
    def export_scenario(filename, sample, sheetsize):
        sizes = []
        branch = create_branch(
            BRANCH_ORIGIN, BRANCH_ANGLE, BRANCH_LENGTH,
            BRANCH_WIDTH, BRANCH_NDOMS
        )
        sizes.extend([
            [cfg.DOMINO_EXTENTS[0], cfg.DOMINO_EXTENTS[1]]
        ] * branch.shape[0])
        left_row = create_wave(
            LEFT_ROW_ORIGIN, LEFT_ROW_ANGLE, LEFT_ROW_LENGTH,
            sample[1], int(sample[0])
        )
        sizes.extend([
            [cfg.DOMINO_EXTENTS[0], cfg.DOMINO_EXTENTS[1]]
        ] * left_row.shape[0])
        lever = [[LEVER_POS.x, LEVER_POS.y, 0]]
        sizes.append([LEVER_EXTENTS[0], LEVER_EXTENTS[1]])
        preplank = [[
            sample[2] + PREPLANK_EXTENTS[0]/2 - .005, PREPLANK_POS.y, 0
        ]]
        sizes.append([PREPLANK_EXTENTS[0], PREPLANK_EXTENTS[1]])
        r_rad = math.radians(sample[4])
        plank = [[
            preplank[0][0] + PREPLANK_EXTENTS[0]/2
            + cfg.PLANK_LENGTH/2*math.cos(r_rad)
            - cfg.PLANK_THICKNESS/2*math.sin(r_rad),
            PLANK_POS.y,
            0
        ]]
        sizes.append([cfg.PLANK_LENGTH*math.cos(r_rad), cfg.PLANK_EXTENTS[1]])
        right_row = create_line(
            RIGHT_ROW_ORIGIN, RIGHT_ROW_ANGLE, RIGHT_ROW_LENGTH,
            RIGHT_ROW_NDOMS
        )
        sizes.extend([
            [cfg.DOMINO_EXTENTS[0], cfg.DOMINO_EXTENTS[1]]
        ] * right_row.shape[0])
        switch = create_x_switch(
            SWITCH_ORIGIN, SWITCH_ANGLE, BRANCH_WIDTH, 11
        )
        sizes.extend([
            [cfg.DOMINO_EXTENTS[0], cfg.DOMINO_EXTENTS[1]]
        ] * switch.shape[0])

        coords = np.vstack(
            [branch, left_row, lever, preplank, plank, right_row, switch]
        )
        sizes = np.array(sizes)

        xy = coords[:, :2] * 100
        xy = xy - (xy.min(axis=0) + xy.max(axis=0))/2 + np.asarray(sheetsize)/2
        a = coords[:, 2]
        sizes *= 100

        vec = VectorFile(filename, sheetsize)
        vec.add_rectangles(xy, a, sizes)
        vec.save()

    def succeeded(self):
        return self.terminate.status == 'success'


class Model:
    """Interface between the optimizer and the objects being optimized."""
    # Bounds
    min_nd = 16
    max_nd = 25
    min_wd = 0.
    max_wd = .1
    min_x = BALL_POS.x
    max_x = BALL_POS.x + cfg.PLANK_LENGTH / 2
    min_z = cfg.BALL_RADIUS + cfg.PLANK_THICKNESS
    max_z = LEVER_HEIGHT - cfg.BALL_RADIUS
    min_r = 0.
    max_r = 75.

    def __init__(self):
        self.scenario = None
        self.left_time = None
        self.right_time = None

    @classmethod
    def get_bounds(cls):
        """Get the bounds of each parameter as a list of (min, max) tuples."""
        return [
            (cls.min_nd, cls.max_nd),
            (cls.min_wd, cls.max_wd),
            (cls.min_x, cls.max_x),
            (cls.min_z, cls.max_z),
            (cls.min_r, cls.max_r),
        ]

    @classmethod
    def get_grid(cls, steps):
        """Get a n-dim grid sampling of the model parameters.

        Parameters
        ----------
        steps : (n_params,) int (real or imaginary) sequence
          Resolution for each parameter/axis of the grid.
           - 0 means no sampling
           - otherwise: interpreted as the 'step' part of a numpy slice,
           meaning that a real integer is a slice step,
           while an imaginary integer is a number of points.

        """
        bounds = cls.get_bounds()
        axes = [np.s_[mi:ma:step]
                for (mi, ma), step in zip(bounds, steps) if step]
        return np.mgrid[axes]

    @classmethod
    def opt2model(cls, X_opt):
        """Map continuous variables from [0, 1] to their original interval."""
        wd = X_opt[0]*(cls.max_wd - cls.min_wd) + cls.min_wd
        x = X_opt[0]*(cls.max_x - cls.min_x) + cls.min_x
        z = X_opt[1]*(cls.max_z - cls.min_z) + cls.min_z
        r = X_opt[2]*(cls.max_r - cls.min_r) + cls.min_r
        return wd, x, z, r

    @classmethod
    def model2opt(cls, X_model):
        """Map continuous variables from their original interval to [0, 1]."""
        wd, x, z, r = X_model
        X = np.array([
            (wd - cls.min_wd) / (cls.max_wd - cls.min_wd),
            (x - cls.min_x) / (cls.max_x - cls.min_x),
            (z - cls.min_z) / (cls.max_z - cls.min_z),
            (r - cls.min_r) / (cls.max_r - cls.min_r),
        ])
        return X

    def update(self, x, run_simu=True, _visual=False):
        if not np.isfinite(x).all():
            self.left_time = math.nan
            self.right_time = math.nan
            return
        if _visual:
            scenario = DominoesBallSync(x, make_geom=True, verbose_cond=True)
            simu = Simulation(scenario, timestep=OPTIM_SIMU_TIMESTEP)
            simu.run_visual()
        # Create new scenario.
        scenario = DominoesBallSync(x)
        self.scenario = scenario
        scene = scenario.scene
        if not run_simu:
            return
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
        if not scenario.succeeded():
            self.left_time = math.nan
            self.right_time = math.nan
            return
        # Update output variables.
        self.left_time = observers[0].times[-1]
        self.right_time = observers[1].times[-1]


class Objective:
    def __init__(self, model, err_val=np.nan):
        self.model = model
        self.err_val = err_val

    def __call__(self, x):
        print("Starting", x)
        self.model.update(x, _visual=0)
        f = self.model.left_time - self.model.right_time
        if not np.isfinite(f):
            f = self.err_val
        print("Internal call to objective -- x = {}, f = {}".format(x, f))
        return f


class NonPenetrationConstraint:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        self.model.update(x, run_simu=False)
        world = self.model.scenario.world
        scene = self.model.scenario.scene
        # Get the bodies at risk of colliding
        left_row_doms = scene.find("left_row*").node().get_children()
        plank = scene.find("plank*").node()
        plank.set_static(False)  # Otherwise we won't detect collision with
        plank.set_active(True)   # the floor. Both are necessary!
        right_row_doms = scene.find("right_row*").node().get_children()
        floor = scene.find("floor*").node()
        targets = right_row_doms + (floor,)
        # Accumulate penetration depths
        total = sum(
            self.get_sum_of_penetration_depths(plank, target, world)
            for target in targets
        ) + sum(
            self.get_sum_of_penetration_depths(d1, d2, world)
            for d1, d2 in zip(left_row_doms[:-1], left_row_doms[1:])
        )
        return total

    @staticmethod
    def get_sum_of_penetration_depths(node0, node1, world):
        penetration = 0
        result = world.contact_test_pair(node0, node1)
        for contact in result.get_contacts():
            mpoint = contact.get_manifold_point()
            penetration += mpoint.get_distance()
        return penetration


def main():
    x_init = [LEFT_ROW_NDOMS, LEFT_ROW_WIDTH,
              BALL_POS.x, BALL_POS.z, PLANK_HPR.z]
    model = Model()
    print("Bounds:", Model.get_bounds())
    # model.update(x_init, _visual=True)
    objective = Objective(model)
    constraints = (
        {'type': 'ineq', 'fun': NonPenetrationConstraint(model)},
    )

    def fromleft(xl):
        return list(xl) + [BALL_POS.x, BALL_POS.z, PLANK_HPR.z]
    # Brute force on the left side.
    grid_left = Model.get_grid(steps=(1, 20j, 0, 0, 0))
    grid_left_vec = grid_left.reshape(grid_left.shape[0], -1).T
    print("Bruteforce:", grid_left_vec.shape[0], "samples")
    filename = "left_side_grid_values.npy"
    try:
        vals = np.load(filename)
    except FileNotFoundError:
        cons = constraints[0]['fun']
        valid = np.array([cons(fromleft(x)) >= 0
                          for x in grid_left_vec], dtype=bool)
        valid.shape = grid_left[0].shape
        print("Validity tests DONE: ", valid.sum(), "valid samples")
        vals = np.empty(valid.shape)
        vals[~valid] = np.nan
        model.scenario = None  # to avoid serialization issues with BAM
        vals[valid] = Parallel(n_jobs=6)(
            delayed(objective)(fromleft(x)) for x in grid_left_vec[valid.flat]
        )
        np.save(filename, vals)
    print(np.isfinite(vals).sum(), "valid samples after simulation")
    if 0:
        fig, ax = plt.subplots()
        ax.hist(vals[np.isfinite(vals)].flat)
        plt.plot()

    rob_left_estimator = DominoRobustness2()

    def get_rob_left(x):
        coords = create_wave(LEFT_ROW_ORIGIN, LEFT_ROW_ANGLE, LEFT_ROW_LENGTH,
                             x[1], int(x[0]))
        return rob_left_estimator(coords).min()
    rob_left = vals.copy()
    valid = np.isfinite(vals)
    rob_left[valid] = [get_rob_left(x) for x in grid_left_vec[valid.flat]]

    good_left = np.logical_and(valid, vals > .1)
    best_left = np.argmax(rob_left[good_left])
    best_left = grid_left_vec[good_left.flat][best_left]
    print("Objective on the left:", objective(fromleft(best_left)))

    if 0:
        fig, ax = plt.subplots()
        x = vals[valid].ravel()
        y = rob_left[valid].ravel()
        c = y * (x > 0)
        ax.scatter(x, y, c=c)
        plt.show()

    if 0:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        extent = [
            grid_left[0, 0, 0], 2*grid_left[0, -1, -1]-grid_left[0, -2, -1],
            grid_left[1, 0, 0], 2*grid_left[1, -1, -1]-grid_left[1, -1, -2]
        ]

        im = ax1.imshow(vals.T, origin='lower', extent=extent, aspect='auto')
        # fig.colorbar(im)
        ax1.scatter(*best_left)
        ax1.set_xlim(*extent[:2])
        ax1.set_ylim(*extent[2:])
        ax1.set_title("Time difference left vs. right side (red=invalid).\n"
                      "Brighter values mean that left is slower than right.")
        ax1.set_xlabel("Number of dominoes")
        ax1.set_ylabel("Amplitude of the domino wave")

        im = ax2.imshow(rob_left.T, origin='lower', extent=extent,
                        aspect='auto')
        ax2.scatter(*best_left)
        ax2.set_xlim(*extent[:2])
        ax2.set_ylim(*extent[2:])
        ax2.set_title("Robustness of the left side.\n"
                      "Brighter values are more robust.")
        ax2.set_xlabel("Number of dominoes")
        ax2.set_ylabel("Amplitude of the domino wave")
        plt.show()

    def fromright(xr):
        return [LEFT_ROW_NDOMS, LEFT_ROW_WIDTH] + list(xr)
    # Brute force on the right side.
    grid_right = Model.get_grid(steps=(0, 0, 20j, 20j, 20j))
    grid_right_vec = grid_right.reshape(grid_right.shape[0], -1).T
    print("Bruteforce:", grid_right_vec.shape[0], "samples")
    filename = "right_side_grid_values.npy"
    try:
        vals = np.load(filename)
    except FileNotFoundError:
        cons = constraints[0]['fun']
        model.scenario = None  # to avoid serialization issues with BAM
        valid = Parallel(n_jobs=6)(
            delayed(cons)(fromright(x)) for x in grid_right_vec
        )
        valid = np.array(valid) >= 0
        valid.shape = grid_right[0].shape
        print("Validity tests DONE: ", valid.sum(), "valid samples")
        vals = np.empty(valid.shape)
        vals[~valid] = np.nan
        vals[valid] = Parallel(n_jobs=6)(
            delayed(objective)(fromright(x))
            for x in grid_right_vec[valid.flat]
        )
        np.save(filename, vals)
    print(np.isfinite(vals).sum(), "valid samples after simulation")
    best_id = np.nanargmax(vals)
    best_right = grid_right_vec[best_id]
    print("Objective on the right:", objective(fromright(best_right)))
    if 0:
        fig, ax = plt.subplots()
        extent = [
            grid_right[0, 0, 0, 0],
            2*grid_right[0, -1, -1, 0]-grid_right[0, -2, -1, 0],
            grid_right[1, 0, 0, 0],
            2*grid_right[1, -1, -1, 0]-grid_right[1, -1, -2, 0]
        ]
        best_id = np.unravel_index(best_id, grid_right[0].shape)
        vals2D = vals[:, :, best_id[-1]]
        im = ax.imshow(vals2D.T, origin='lower', extent=extent, aspect='auto')
        fig.colorbar(im)
        ax.scatter(*best_right)
        ax.set_xlim(*extent[:2])
        ax.set_ylim(*extent[2:])
        ax.set_title("Time difference left vs. right side (red=invalid).\n"
                     "Higher values mean that left is slower than right.")
        ax.set_xlabel("Ball x")
        ax.set_ylabel("Ball z")
        plt.show()

    x_best = np.concatenate([best_left, best_right])

    if 0:
        bounds = np.column_stack((x_best*0.9, x_best*1.1))
        print(bounds)
        objective.err_val = 2
        x_init = x_best
        x_best = opt.minimize(
            objective, x_init, bounds=bounds,
            # constraints=constraints,
            method='SLSQP',
            options=dict(disp=True, maxiter=10, eps=1e-4)
        ).x
        print(x_best)
        print(objective(x_best))

    # Show the solution.
    print("Constraint:", constraints[0]['fun'](x_best))
    print("Overall objective:", objective(x_best))
    model.update(x_best, _visual=True)

    if 0:
        # Export the solution.
        DominoesBallSync.export_scenario("base.pdf", x_init, (5*21, 1*29.7))
        DominoesBallSync.export_scenario("best.pdf", x_best, (5*21, 1*29.7))


if __name__ == "__main__":
    main()
