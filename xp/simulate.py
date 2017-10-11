"""
This module provides the necessary functions to simulate a domino run.

"""
from panda3d.bullet import BulletWorld, BulletBoxShape, BulletRigidBodyNode
from panda3d.core import load_prc_file_data
from panda3d.core import NodePath
from panda3d.core import Point3, Vec3


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


def setup_dominoes(global_coords, _add_geom=False):
    """Generate the objects used in the domino run simulation.

    Parameters
    ----------
    global_coords : (n,3) array
      Sequence of n global coordinates (x, y, heading) for each domino.
    _add_geom : bool, optional
      Whether to generate a geometry for each domino. Used for debug.

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

    return doms_np, world


def run_simu(doms_np, world, _visual=False):
    """Run the domino run simulation.

    Parameters
    ----------
    doms_np : NodePath
      NodePath containing the dominoes.
    world : BulletWorld
      BulletWorld used for the simulation.
    _visual : bool, optional
      In visual mode, a Panda3D app is opened to show the simulation. Times
      are not recorded (an empty list is returned).

    Returns
    -------
    top_times : (n,) sequence
      The toppling time of each domino. A domino that doesn't topple has
      a time equal to np.inf.

    """
    return top_times


