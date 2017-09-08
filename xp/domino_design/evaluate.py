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
from sklearn.externals.joblib import Parallel

from config import t, w, h, density, NCORES
sys.path.insert(0, os.path.abspath("../.."))
from primitives import DominoMaker
from primitives import Floor
import spline2d as spl
sys.path.insert(0, os.path.abspath(".."))
from domino_learning.functions import tilt_box_forward


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


def get_overlapping_dominoes(u, spline):
    """Get the indices of the dominoes that overlap."""
    overlapping = []
    if len(u) < 2:
        return overlapping
    base = box(-t/2, -w/2, t/2,  w/2)
    x, y = spl.splev(u, spline)
    a = spl.splang(u, spline)
    dominoes = [translate(rotate(base, ai), xi, yi)
                for xi, yi, ai in zip(x, y, a)]
    # Not efficient but who cares
    for i, d1 in enumerate(dominoes):
        for d2 in dominoes:
            if d2 is not d1 and d1.intersects(d2):
                overlapping.append(i)
                break

    return overlapping


def test_no_overlap(u, spline):
    return not bool(get_overlapping_dominoes(u, spline))


def test_no_overlap_fast(u, spline):
    """Only test for overlaps between successive dominoes."""
    if len(u) < 2:
        return True
    base = box(-t/2, -w/2, t/2,  w/2)
    x, y = spl.splev(u, spline)
    a = spl.splang(u, spline)
    b1 = None
    b2 = translate(rotate(base, a[0]), x[0], y[0])
    for i in range(1, len(u)):
        b1 = b2
        b2 = translate(rotate(base, a[i]), x[i], y[i])
        if b1.intersects(b2):
            return False
    return True


def get_toppling_angle():
    return math.atan(t / h) * 180 / math.pi + 1




def setup_dominoes(u, spline):
    """Setup the world and objects to run the simulation.

    Parameters
    ----------
    u : sequence
        Samples along the spline.
    spline : 'spline' as defined in spline2d
        Path of the domino run.

    Returns
    -------
    doms_np : NodePath
        Contains all the dominoes.
    world : BulletWorld
        World for the simulation.
    """
    # World
    world = BulletWorld()
    world.set_gravity((0, 0, -9.81))
    # Floor
    floor_np = NodePath("floor")
    floor = Floor(floor_np, world)
    floor.create()
    # Dominoes
    doms_np = NodePath("domino_run")
    domino_factory = DominoMaker(doms_np, world, make_geom=False)
    u = np.asarray(u)
    positions = spl.splev3d(u, spline, h/2)
    headings = spl.splang(u, spline)
    mass = density * t * w * h
    extents = Vec3(t, w, h)
    for i, (pos, head) in enumerate(zip(positions, headings)):
        domino_factory.add_domino(
                Vec3(*pos), head, extents, mass, prefix="domino_{}".format(i))
    # Set initial conditions for first domino
    first_domino = doms_np.get_child(0)
    #  angvel_init = Vec3(0., 15., 0.)
    #  angvel_init = Mat3.rotate_mat(spl.splang(0, spline)).xform(
            #  angvel_init)
    #  first_domino.node().set_angular_velocity(angvel_init)
    toppling_angle = get_toppling_angle()
    tilt_box_forward(first_domino, toppling_angle)
    first_domino.node().set_transform_dirty()

    return doms_np, world


def run_simu(doms_np, world):
    """Run the simulation.

    Parameters
    ----------
    doms_np : NodePath
        Contains all the dominoes.
    world : BulletWorld
        World for the simulation.

    """
    time = 0.
    n = doms_np.get_num_children()
    maxtime = n
    toppling_angle = get_toppling_angle()
    last_domino = doms_np.get_child(n - 1)
    while (last_domino.get_r() < toppling_angle
            and any(dom.node().is_active() for dom in doms_np.get_children())
            and time < maxtime):
        time += 1/120
        world.do_physics(1/120, 2, 1/120)
    return doms_np


def get_toppling_fraction(u, spline, doms_np):
    toppling_angle = get_toppling_angle()
    n = doms_np.get_num_children()
    try:
        # Find the first domino that didn't topple.
        idx = next(i for i in range(n)
                   if doms_np.get_child(i).get_r() < toppling_angle)
    except StopIteration:
        # All dominoes toppled
        idx = n
    return spl.arclength(spline, u[idx-1]) / spl.arclength(spline)


def test_all_toppled(doms_np):
    toppling_angle = get_toppling_angle()
    return all(dom.get_r() >= toppling_angle for dom in doms_np.get_children())


def evaluate_domino_run(u, spline):
    """Evaluate the domino distribution for all criteria."""
    covered = test_path_coverage(u, spline)
    non_overlapping = test_no_overlap(u, spline)
    doms_np, world = setup_dominoes(u, spline)
    doms_np = run_simu(doms_np, world)
    all_topple = test_all_toppled(doms_np)
    top_frac = get_toppling_fraction(u, spline, doms_np)

    results = [covered, non_overlapping, all_topple, top_frac]
    return results


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
            delayed(evaluate_domino_run)(domruns['arr_{}'.format(i)], s)
            for i, s in enumerate(splines))
    #  results = [evaluate_domino_run(domruns['arr_{}'.format(i)], s)
    #         for i, s in enumerate(splines)]

    dirname = os.path.dirname(dompath)
    prefix = os.path.splitext(os.path.basename(dompath))[0]
    outname = prefix + "-validity.npy"
    np.save(os.path.join(dirname, outname), results)


if __name__ == "__main__":
    main()
