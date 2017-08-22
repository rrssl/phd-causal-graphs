"""
Evaluate whether the runs are successful or not.

Parameters
----------
splpath : string
  Path to the list of splines
dompath : string
  Path to the domino runs.

"""
import math
import os
import pickle
import sys

import numpy as np
from panda3d.bullet import BulletWorld
from panda3d.core import NodePath
from panda3d.core import Vec3
from panda3d.core import Mat3
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box

from config import t, w, h
sys.path.insert(0, os.path.abspath("../.."))
from primitives import DominoMaker
from primitives import Floor
import spline2d as spl


def test_path_coverage(u, spline):
    return (spl.arclength(spline) - spl.arclength(spline, u[-1])) < h


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


def test_all_topple(u, spline):
    if len(u) < 2:
        return True
    u = np.array(u)
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
    for i, (pos, head) in enumerate(zip(positions, headings)):
        domino_factory.add_domino(
                Vec3(*pos), head, Vec3(t, w, h), mass=1,
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
    toppling_angle = math.atan(t / h) * 180 / math.pi + 1
    while last_domino.get_r() < toppling_angle and time < maxtime:
        time += 1/60
        world.do_physics(1/60, 2)

    return last_domino.get_r() >= toppling_angle


def test_domino_run(u, spline):
    """Test if the path is filled, there is no overlap, and all dominoes
    topple."""
    return bool(test_path_coverage(u, spline) and
                test_no_overlap(u, spline) and
                test_all_topple(u, spline)
                )


def main():
    if len(sys.argv) < 3:
        print("Please the location of the two necessary files.")
        return
    splpath = sys.argv[1]
    dompath = sys.argv[2]

    with open(splpath, 'rb') as fs:
        splines = pickle.load(fs)
    with open(dompath, 'rb') as fd:
        domruns = pickle.load(fd)

    results = [test_domino_run(u, s) for u, s in zip(domruns, splines)]
    print(results)

    with open("validity-method_{}.pkl".format(dompath[-5]), 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    main()
