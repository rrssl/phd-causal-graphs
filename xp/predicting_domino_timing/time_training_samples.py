"""
Run the simulation for each training sample and reccord the
toppling-to-toppling time.

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

sys.path.insert(0, os.path.abspath(".."))
from predicting_domino_timing.config import t, w, h, density
from predicting_domino_timing.config import timestep, MAX_WAIT_TIME
from predicting_domino_timing.config import NCORES
from predicting_domino_timing.config import NPREV
from domino_design.evaluate import get_toppling_angle
from predicting_domino_toppling.functions import tilt_box_forward

sys.path.insert(0, os.path.abspath("../.."))
from primitives import Floor, DominoMaker


# See ../predicting_domino_toppling/functions.py
load_prc_file_data("", "garbage-collect-states 0")


def run_predicting_domino_timing_xp(x, y, a, s):
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
    m = density * t * w * h
    length = s * NPREV
    x = np.concatenate((np.linspace(-length, 0, NPREV), [x]))
    y = np.concatenate((np.zeros(NPREV), [y]))
    a = np.concatenate((np.zeros(NPREV), [a]))
    for i, (xi, yi, ai) in enumerate(zip(x, y, a)):
        dom_fact.add_domino(Vec3(xi, yi, h*.5), ai, Vec3(t, w, h), m,
                            "d{}".format(i))
    d0 = dom_path.get_child(0)
    dpen = dom_path.get_child(NPREV-1)
    dlast = dom_path.get_child(NPREV)
    # Initial state
    toppling_angle = get_toppling_angle()
    tilt_box_forward(d0, toppling_angle)
    d0.node().set_transform_dirty()

    time_prev = 0
    time = 0
    maxtime = (NPREV+1) * MAX_WAIT_TIME
    while time < maxtime:
        if dpen.get_r() >= toppling_angle and time_prev == 0:
            time_prev = time
        # Early termination conditions
        if dlast.get_r() >= toppling_angle:
            return time - time_prev
        if time_prev > 0 and (time - time_prev) > MAX_WAIT_TIME:
            return np.inf
        if not any(di.node().is_active() for di in dom_path.get_children()):
            return np.inf

        time += timestep
        world.do_physics(timestep, 2, timestep)
    else:
        return np.inf


def compute_times(samples):
    times = Parallel(n_jobs=NCORES)(
            delayed(run_predicting_domino_timing_xp)(x, y, a, s)
            for x, y, a, s in samples)
    return times


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    spath = sys.argv[1]
    samples = np.load(spath)
    assert samples.shape[1] == 4, "Number of dimensions must be 4."
    times = compute_times(samples)
    root, _ = os.path.splitext(spath)
    np.save(root + "-times.npy", times)


if __name__ == "__main__":
    main()
