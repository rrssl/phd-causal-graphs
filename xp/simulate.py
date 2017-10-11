"""
This module provides the necessary functions to simulate a domino run.

"""
from pathlib import Path
import sys

import numpy as np

from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.core import load_prc_file_data
from panda3d.core import NodePath
from panda3d.core import Point3, Vec3

from .config import t, w, h, MASS, TOPPLING_ANGLE
from .config import TIMESTEP, MAX_WAIT_TIME

sys.path.insert(0, str(Path(__file__).resolve().parent))
from primitives import DominoMaker, Floor


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


def rotate_around(path: NodePath, pivot: Point3, hpr: Vec3):
    """Rotate in place the NodePath around a 3D point.

    Parameters
    ----------
    path : NodePath
        The NodePath to rotate.
    pivot : Point3
        Center of rotation, in the relative to the NodePath.
    hpr : Vec3
        HPR components of the rotation, relative to the NodePath.
    """
    pivot_np = NodePath("pivot")
    pivot_np.set_pos(path, pivot)
    pivot_np.set_hpr(path.get_hpr())  # Put pivot in the same relative frame
    path.set_hpr(pivot_np, hpr)


def tilt_box_forward(box: NodePath, angle):
    extents = box.node().get_shape(0).get_half_extents_with_margin()
    ctr = Point3(extents[0], 0, -extents[2])
    rotate_around(box, ctr, Vec3(0, 0, angle))


def setup_dominoes(global_coords, _make_geom=False):
    """Generate the objects used in the domino run simulation.

    Parameters
    ----------
    global_coords : (n,3) array
      Sequence of n global coordinates (x, y, heading) for each domino.

    Returns
    -------
    doms_np : NodePath
      NodePath containing the dominoes.
    world : BulletWorld
      BulletWorld used for the simulation.

    """
    # World
    world = BulletWorld()
    world.set_gravity(Vec3(0, 0, -9.81))
    # Floor
    floor_path = NodePath("floor")
    floor = Floor(floor_path, world)
    floor.create()
    # Dominoes
    doms_np = NodePath("domino_run")
    domino_factory = DominoMaker(doms_np, world, make_geom=_make_geom)
    extents = Vec3(t, w, h)
    for i, (x, y, a) in enumerate(global_coords):
        domino_factory.add_domino(
                Vec3(x, y, h/2), a, extents, MASS, prefix="D{}".format(i))
    # Set initial conditions for first domino
    first_domino = doms_np.get_child(0)
    tilt_box_forward(first_domino, TOPPLING_ANGLE+1)
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
        import os, sys
        sys.path.insert(0, os.path.abspath('..'))
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
        if dominoes[last_toppled_id+1].get_r() >= TOPPLING_ANGLE:
            last_toppled_id += 1
            toppling_times[last_toppled_id] = time
        if last_toppled_id == n-1:
            # All dominoes toppled in order.
            break
        if dominoes[last_toppled_id+1].get_r() >= TOPPLING_ANGLE:
            print("Warning: next domino had already toppled")
        if time - toppling_times[last_toppled_id] > MAX_WAIT_TIME:
            # The chain broke
            break
        time += timestep
        world.do_physics(timestep, 2, timestep)
    return toppling_times
