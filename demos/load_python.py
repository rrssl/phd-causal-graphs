"""
Load, simulate and visualize a scenario instance defined as a python file.

Parameters
----------
path : string
  Path to the python file.

"""
import importlib.util
import os
import sys
import tempfile

from panda3d.core import load_prc_file_data

sys.path.insert(0, os.path.abspath(".."))
from core.scenario import (StateObserver, load_scenario,  # noqa: E402
                           simulate_scene)
from gui.viewers import Replayer  # noqa: E402

FPS = 600
DURATION = 2


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    script = load_module("scene_script", path)
    DATA = script.DATA
    dir_ = tempfile.mkdtemp()
    # Create the scene geometry.
    scenario = load_scenario(DATA, geom='HD', phys=False)
    scene_path = os.path.join(dir_, "scene")
    scenario.scene.export_scene_to_egg(scene_path)
    # Run the scenario.
    scenario = load_scenario(DATA)
    obs = StateObserver(scenario.scene)
    simulate_scene(scenario.scene, duration=DURATION, timestep=1/FPS,
                   callbacks=[obs])
    simu_path = os.path.join(dir_, "simu.pkl")
    obs.export(simu_path, fps=FPS)
    # Show the simulation.
    load_prc_file_data("", "win-origin 500 200")
    app = Replayer(scene_path+".bam", simu_path)
    app.cam_distance = 1
    app.min_cam_distance = .01
    app.camLens.set_near(.01)
    app.zoom_speed = .01
    app.run()


if __name__ == "__main__":
    main()
