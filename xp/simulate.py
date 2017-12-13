"""
This module provides the necessary functions to simulate a domino run.

"""
import os
import random
import sys

import numpy as np

from panda3d.bullet import BulletWorld
from panda3d.core import load_prc_file_data
from panda3d.core import NodePath
from panda3d.core import Vec3
from shapely.geometry import box
from shapely.affinity import rotate, translate

from .config import (t, w, h, MASS, FLOOR_MATERIAL_FRICTION,
                     DOMINO_MATERIAL_FRICTION, DOMINO_MATERIAL_RESTITUTION,
                     DOMINO_ANGULAR_DAMPING, TOPPLING_ANGLE,
                     X_MAX, Y_MAX, A_MAX, TIMESTEP, MAX_WAIT_TIME)
from .domgeom import tilt_box_forward

sys.path.insert(0, os.path.abspath(".."))
from primitives import DominoMaker, Floor  # noqa
import spline2d as spl  # noqa


# The next line avoids a "memory leak" that notably happens when
# BulletWorld.do_physics is called a huge number of times out of the
# regular Panda3D task process. In a nutshell, objects transforms are
# cached and compared by pointer to avoid expensive recomputation; the
# cache is configured to flush itself at the end of each frame, which never
# happens when we don't use frames. The solutions are: don't use the cache
# ("transform-cache 0"), or don't defer flushing to the end of the frame
# ("garbage-collect-states 0"). See
# http://www.panda3d.org/forums/viewtopic.php?t=15645 for a discussion.
load_prc_file_data("", "garbage-collect-states 0")


def perturbate_dominoes(coords, randfactor, maxtrials=10):
    """Randomly perturbate dominoes, but keep the distribution valid.

    Parameters
    ----------
    coords : (n,3) numpy array
        Global X, Y and heading of each domino.
    randfactor : float
        Randomization factor: fraction of the 'reasonable' range for each
        coordinate, which will be used as the parameter of the uniform
        distribution used for the randomization.

    Returns
    -------
    new_coords : (n,3) numpy array
        New coordinates.

    """
    base = box(-t/2, -w/2, t/2,  w/2)
    dominoes = [translate(rotate(base, ai), xi, yi) for xi, yi, ai in coords]
    new_coords = np.empty_like(coords)
    rng_x = X_MAX * randfactor
    rng_y = Y_MAX * randfactor
    rng_a = A_MAX * randfactor
    for i in range(len(dominoes)):
        ntrials = 0
        while ntrials < maxtrials:
            new_coords[i] = coords[i] + [random.uniform(-rng_x, rng_x),
                                         random.uniform(-rng_y, rng_y),
                                         random.uniform(-rng_a, rng_a)]
            dominoes[i] = translate(
                    rotate(base, new_coords[i, 2]), *new_coords[i, :2])

            # Find first domino to intersect the current domino
            try:
                next(dom for dom in dominoes
                     if dom is not dominoes[i] and dom.intersects(dominoes[i]))
            except StopIteration:
                # No domino intersects the current one
                break
            ntrials += 1
        else:
            # Valid perturbated coordinates could not be found in time.
            new_coords[i] = coords[i]
            dominoes[i] = translate(
                    rotate(base, new_coords[i, 2]), *new_coords[i, :2])
    return new_coords


class Simulation:

    def __init__(self, timestep=TIMESTEP, visual=False):
        self.timestep = timestep
        self._visual = visual
        # World
        self.world = BulletWorld()
        self.world.set_gravity(Vec3(0, 0, -9.81))
        # Floor
        self.floor_path = NodePath("floor")
        floor = Floor(self.floor_path, self.world, make_geom=False)
        floor.create()
        self.floor_path.find("floor_solid").node().set_friction(
                FLOOR_MATERIAL_FRICTION)
        # Domino runs
        self.domino_runs_paths = []

    def run(self):
        """Run the domino run simulation.

        Returns
        -------
        toppling_times : (n,) ndarray
          The toppling time of each domino. A domino that doesn't topple has
          a time equal to np.inf.

        """
        ts = self.timestep
        dominoes = [d
                    for drp in self.domino_runs_paths
                    for d in drp.get_children()]
        n = len(dominoes)
        last_toppled_id = -1
        toppling_times = np.full(n, np.inf)
        time = 0.
        while True:
            # Use while here because close dominoes can topple at the same time
            while (last_toppled_id < n-1
                    and dominoes[last_toppled_id+1].get_r() >= TOPPLING_ANGLE):
                last_toppled_id += 1
                toppling_times[last_toppled_id] = time
            if last_toppled_id == n-1:
                # All dominoes toppled in order.
                break
            if time - toppling_times[last_toppled_id] > MAX_WAIT_TIME:
                # The chain broke
                break
            time += ts
            self.world.do_physics(ts, 2, ts)
        return toppling_times

    def add_domino_run(self, coords, randfactor=0, tilt_first_dom=True):
        """Add a domino run to the simulation.

        Parameters
        ----------
        coords : (n,3) array
          Sequence of n global coordinates (x, y, heading) for each domino.
        randfactor : float
          If > 0, a randomization will be applied to all dominoes' coordinates.
        tilt_first_dom : bool
          Whether or not to tilt the first domino.

        """
        doms_np = NodePath(
                "domino_run_{}".format(len(self.domino_runs_paths)+1))
        self.domino_runs_paths.append(doms_np)
        domino_factory = DominoMaker(
                doms_np, self.world, make_geom=self._visual)
        extents = Vec3(t, w, h)
        for i, (x, y, a) in enumerate(coords):
            dom = domino_factory.add_domino(
                    Vec3(x, y, h/2), a, extents, MASS, prefix="D{}".format(i)
                    ).node()
            dom.set_friction(DOMINO_MATERIAL_FRICTION)
            dom.set_restitution(DOMINO_MATERIAL_RESTITUTION)
            dom.set_angular_damping(DOMINO_ANGULAR_DAMPING)

        if tilt_first_dom:
            # Set initial conditions for first domino
            first_domino = doms_np.get_child(0)
            tilt_box_forward(first_domino, TOPPLING_ANGLE+.5)
            first_domino.node().set_transform_dirty()
            # Alternative with initial velocity:
            #  angvel_init = Vec3(0., 15., 0.)
            #  angvel_init = Mat3.rotate_mat(headings[0]).xform(angvel_init)
            #  first_domino.node().set_angular_velocity(angvel_init)


def setup_dominoes(coords, randfactor=0, tilt_first_dom=True,
                   _make_geom=False):
    """Generate the objects used in the domino run simulation.

    Parameters
    ----------
    coords : (n,3) array
      Sequence of n global coordinates (x, y, heading) for each domino.
    randfactor : float
      If > 0, a randomization will be applied to all dominoes' coordinates.
    tilt_first_dom : bool
      Whether or not to tilt the first domino.

    Returns
    -------
    doms_np : NodePath
      NodePath containing the dominoes.
    world : BulletWorld
      BulletWorld used for the simulation.

    """
    if randfactor:
        coords = perturbate_dominoes(coords, randfactor)
    # World
    world = BulletWorld()
    world.set_gravity(Vec3(0, 0, -9.81))
    # Floor
    floor_path = NodePath("floor")
    floor = Floor(floor_path, world, make_geom=_make_geom)
    floor.create()
    floor_path.find("floor_solid").node().set_friction(FLOOR_MATERIAL_FRICTION)
    # Dominoes
    doms_np = NodePath("domino_run")
    domino_factory = DominoMaker(doms_np, world, make_geom=_make_geom)
    extents = Vec3(t, w, h)
    for i, (x, y, a) in enumerate(coords):
        dom = domino_factory.add_domino(
                Vec3(x, y, h/2), a, extents, MASS, prefix="D{}".format(i)
                ).node()
        dom.set_friction(DOMINO_MATERIAL_FRICTION)
        dom.set_restitution(DOMINO_MATERIAL_RESTITUTION)
        dom.set_angular_damping(DOMINO_ANGULAR_DAMPING)

    if tilt_first_dom:
        # Set initial conditions for first domino
        first_domino = doms_np.get_child(0)
        tilt_box_forward(first_domino, TOPPLING_ANGLE+.5)
        first_domino.node().set_transform_dirty()
        # Alternative with initial velocity:
        #  angvel_init = Vec3(0., 15., 0.)
        #  angvel_init = Mat3.rotate_mat(headings[0]).xform(angvel_init)
        #  first_domino.node().set_angular_velocity(angvel_init)

    return doms_np, world


def run_simu(doms_np: NodePath, world: BulletWorld, timestep=TIMESTEP,
             _visual=False):
    """Run the domino run simulation.

    Parameters
    ----------
    doms_np : NodePath
      NodePath containing the dominoes.
    world : BulletWorld
      BulletWorld used for the simulation.

    Returns
    -------
    toppling_times : (n,) ndarray
      The toppling time of each domino. A domino that doesn't topple has
      a time equal to np.inf.

    """
    if _visual:
        from viewers import PhysicsViewer

        app = PhysicsViewer()
        doms_np.reparent_to(app.models)
        app.world = world
        try:
            app.run()
        except SystemExit:
            app.destroy()
        return

    dominoes = list(doms_np.get_children())
    n = len(dominoes)
    last_toppled_id = -1
    toppling_times = np.full(n, np.inf)
    time = 0.
    while True:
        # Use a while here because close dominoes can topple at the same time
        while (last_toppled_id < n-1
                and dominoes[last_toppled_id+1].get_r() >= TOPPLING_ANGLE):
            last_toppled_id += 1
            toppling_times[last_toppled_id] = time
        if last_toppled_id == n-1:
            # All dominoes toppled in order.
            break
        if time - toppling_times[last_toppled_id] > MAX_WAIT_TIME:
            # The chain broke
            break
        time += timestep
        world.do_physics(timestep, 2, timestep)
    return toppling_times


def setup_dominoes_from_path(u, spline, randfactor=0, tilt_first_dom=True,
                             _make_geom=False):
    """Setup the world and objects to run the simulation.

    Parameters
    ----------
    u : sequence
        Samples along the spline.
    spline : 'spline' as defined in spline2d
        Path of the domino run.
    randfactor : float
      If > 0, a randomization will be applied to all dominoes' coordinates.
    tilt_first_dom : bool
      Whether or not to tilt the first domino.

    Returns
    -------
    doms_np : NodePath
        Contains all the dominoes.
    world : BulletWorld
        World for the simulation.
    """
    u = np.asarray(u)
    x, y = spl.splev(u, spline)
    a = spl.splang(u, spline)
    coords = np.column_stack((x, y, a))

    return setup_dominoes(coords, randfactor, tilt_first_dom, _make_geom)
