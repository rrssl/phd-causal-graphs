"""
Run the simulation for each training sample and record the topple-to-topple
time.

Parameters
----------
spath : string
  Path to the .npy samples.
nprev : int
  Number of dominoes to place before the pair of interest.

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
from domino_design.evaluate import get_toppling_angle
from predicting_domino_toppling.functions import tilt_box_forward

sys.path.insert(0, os.path.abspath("../.."))
from primitives import Floor, DominoMaker


# See ../predicting_domino_toppling/functions.py
load_prc_file_data("", "garbage-collect-states 0")


def compute_toppling_time(x, y, a, s, nprev, _visual=False):
    # World
    world = BulletWorld()
    world.set_gravity(Vec3(0, 0, -9.81))
    # Floor
    floor_path = NodePath("floor")
    floor = Floor(floor_path, world)
    floor.create()
    # Dominoes
    dom_path = NodePath("dominoes")
    dom_fact = DominoMaker(dom_path, world, make_geom=_visual)
    m = density * t * w * h
    length = s * nprev
    x = np.concatenate((np.linspace(-length, 0, nprev), [x]))
    y = np.concatenate((np.zeros(nprev), [y]))
    a = np.concatenate((np.zeros(nprev), [a]))
    for i, (xi, yi, ai) in enumerate(zip(x, y, a)):
        dom_fact.add_domino(Vec3(xi, yi, h*.5), ai, Vec3(t, w, h), m,
                            "d{}".format(i))
    d0 = dom_path.get_child(0)
    dpen = dom_path.get_child(nprev-1)
    dlast = dom_path.get_child(nprev)
    # Initial state
    toppling_angle = get_toppling_angle()
    tilt_box_forward(d0, toppling_angle)
    d0.node().set_transform_dirty()

    if _visual:
        from viewers import PhysicsViewer
        app = PhysicsViewer()
        dom_path.reparent_to(app.models)
        app.world = world
        try:
            app.run()
        except SystemExit:
            app.destroy()
        return

    time_prev = 0
    time = 0
    maxtime = (nprev+1) * MAX_WAIT_TIME
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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    spath = sys.argv[1]
    nprev = int(sys.argv[2])

    samples = np.load(spath)
    assert samples.shape[1] == 4, "Number of dimensions must be 4."
    times = Parallel(n_jobs=NCORES)(
            delayed(compute_toppling_time)(x, y, a, s, nprev)
            for x, y, a, s in samples)

    root, _ = os.path.splitext(spath)
    np.save(root + "-times.npy", times)


if __name__ == "__main__":
    main()
