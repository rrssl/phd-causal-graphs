"""
Evaluate whether the runs are successful or not.

Parameters
----------
splpath : string
  Path to the list of splines
dompath : string
  Path to the domino runs.
ns : int, optional
  Only process the ns first splines of the list.

"""
import math
import os
import pickle
import sys
from tempfile import mkdtemp

import numpy as np
from panda3d.core import load_prc_file_data
from panda3d.bullet import BulletWorld
from panda3d.core import NodePath
from panda3d.core import Vec3
from panda3d.core import Mat3
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Memory
from sklearn.externals.joblib import Parallel

from config import t, w, h, density, NCORES
sys.path.insert(0, os.path.abspath("../.."))
from primitives import DominoMaker
from primitives import Floor
import spline2d as spl


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


def test_path_coverage(u, spline):
    return bool((spl.arclength(spline) - spl.arclength(spline, u[-1])) < h)


def test_no_overlap(u, spline):
    if len(u) < 2:
        return True
    base = box(-t * .5, -w * .5, t * .5,  w * .5)
    x, y = spl.splev(u, spline)
    h = spl.splang(u, spline)
    for i in range(len(u) - 1):
        b1 = translate(rotate(base, h[i]), x[i], y[i])
        b2 = translate(rotate(base, h[i+1]), x[i+1], y[i+1])
        if b1.intersection(b2).area > 0:
            return False
    return True


def get_toppling_angle():
    return math.atan(t / h) * 180 / math.pi + 1


# Memoize calls to run_simu
cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=0)

@memory.cache
def run_simu(u, spline):
    if len(u) < 2:
        return True
    u = np.asarray(u)
    # World
    world = BulletWorld()
    world.set_gravity((0, 0, -9.81))
    # Floor
    floor_path = NodePath("floor")
    floor = Floor(floor_path, world)
    floor.create()
    # Dominoes
    run_np = NodePath("domino_run")
    domino_factory = DominoMaker(run_np, world, make_geom=False)
    positions = spl.splev3d(u, spline, .5*h)
    headings = spl.splang(u, spline)
    mass = density * t * w * h
    for i, (pos, head) in enumerate(zip(positions, headings)):
        domino_factory.add_domino(
                Vec3(*pos), head, Vec3(t, w, h), mass=mass,
                prefix="domino_{}".format(i))
    # Set initial angular velocity
    # (but maybe we should just topple instead of giving velocity)
    angvel_init = Vec3(0., 15., 0.)
    angvel_init = Mat3.rotate_mat(spl.splang(0, spline)).xform(
            angvel_init)
    first_domino = run_np.get_child(0)
    first_domino.node().set_angular_velocity(angvel_init)
    last_domino = run_np.get_child(run_np.get_num_children() - 1)

    time = 0.
    maxtime = len(u)
    toppling_angle = get_toppling_angle()
    while (last_domino.get_r() < toppling_angle
            and any(dom.node().is_active() for dom in run_np.get_children())
            and time < maxtime):
        time += 1/60
        world.do_physics(1/60, 2)

    return run_np


def get_toppling_fraction(u, spline):
    run_np = run_simu(u, spline)
    toppling_angle = get_toppling_angle()
    n = run_np.get_num_children()
    i = 0
    while i < n and run_np.get_child(i).get_r() >= toppling_angle:
        i += 1
    return spl.arclength(spline, u[i-1]) / spl.arclength(spline)


def test_all_topple(u, spline):
    run_np = run_simu(u, spline)
    toppling_angle = get_toppling_angle()
    return all(dom.get_r() >= toppling_angle for dom in run_np.get_children())


def test_domino_run(u, spline):
    """Test if the path is filled, there is no overlap, and all dominoes
    topple."""
    return [test_path_coverage(u, spline),
            test_no_overlap(u, spline),
            test_all_topple(u, spline),
            get_toppling_fraction(u, spline),
            ]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    splpath = sys.argv[1]
    dompath = sys.argv[2]
    ns = int(sys.argv[3]) if len(sys.argv) == 4 else None

    with open(splpath, 'rb') as fs:
        splines = pickle.load(fs)[slice(ns)]
    domruns = np.load(dompath)

    results = Parallel(n_jobs=NCORES)(
            delayed(test_domino_run)(domruns['arr_{}'.format(i)], s)
            for i, s in enumerate(splines))
    #  results = [test_domino_run(domruns['arr_{}'.format(i)], s)
               #  for i, s in enumerate(splines)]

    dirname = os.path.dirname(dompath)
    prefix = os.path.splitext(os.path.basename(dompath))[0]
    outname = prefix + "-validity.npy"
    np.save(os.path.join(dirname, outname), results)


if __name__ == "__main__":
    main()
