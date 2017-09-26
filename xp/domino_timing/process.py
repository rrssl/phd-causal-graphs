"""
Run the simulation for each domino pair and record the toppling-to-toppling
time.

Parameters
----------
spath : string
  Path to the .npy samples.

"""
import os
import sys

import numpy as np
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel

from panda3d.bullet import BulletWorld
from panda3d.core import load_prc_file_data
from panda3d.core import NodePath
from panda3d.core import Vec3

from config import t, w, h, density, timestep, NCORES, MAX_WAIT_TIME
sys.path.insert(0, os.path.abspath(".."))
from domino_design.evaluate import get_toppling_angle
from predicting_domino_toppling.functions import tilt_box_forward

sys.path.insert(0, os.path.abspath("../.."))
from primitives import Floor, DominoMaker


# See ../predicting_domino_toppling/functions.py
load_prc_file_data("", "garbage-collect-states 0")


def run_domino_timing_xp(params, timestep):
    # World
    world = BulletWorld()
    world.set_gravity(Vec3(0, 0, -9.81))
    # Floor
    floor_path = NodePath("floor")
    floor = Floor(floor_path, world)
    floor.create()
    # Dominoes
    dom_path = NodePath("dominoes")
    dom_fact = DominoMaker(dom_path, world, make_geom=False)
    t, w, h, x, y, a, m = params
    d1 = dom_fact.add_domino(Vec3(0, 0, h*.5), 0, Vec3(t, w, h), m, "d1")
    d2 = dom_fact.add_domino(Vec3(x, y, h*.5), a, Vec3(t, w, h), m, "d2")
    # Initial state
    toppling_angle = get_toppling_angle()
    tilt_box_forward(d1, toppling_angle)
    d1.node().set_transform_dirty()

    test = world.contact_test_pair(d1.node(), d2.node())
    if test.get_num_contacts() > 0:
        return np.inf

    time = 0.
    while time < MAX_WAIT_TIME:
        # Early termination conditions
        if d2.get_r() >= toppling_angle:
            return time
        if not (d1.node().is_active() or d2.node().is_active()):
            return np.inf
        time += timestep
        world.do_physics(timestep, 2, timestep)
    else:
        return np.inf


def compute_times(samples):
    m = density * t * w * h
    if samples.shape[1] == 2:
        times = Parallel(n_jobs=NCORES)(
                delayed(run_domino_timing_xp)
                ((t, w, h, d, 0, a, m), timestep)
                for d, a in samples)
    else:
        times = Parallel(n_jobs=NCORES)(
                delayed(run_domino_timing_xp)
                ((t, w, h, x, y, a, m), timestep)
                for x, y, a in samples)
    return times


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    spath = sys.argv[1]
    samples = np.load(spath)
    assert samples.shape[1] in (2, 3), "Number of dimensions must be 2 or 3."
    times = compute_times(samples)
    root, _ = os.path.splitext(spath)
    np.save(root + "-times.npy", times)


if __name__ == "__main__":
    main()
