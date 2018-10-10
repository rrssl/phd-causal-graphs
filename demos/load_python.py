"""
Load, simulate and visualize a scenario instance defined as a python file.

Parameters
----------
path : string
  Path to the python file.
interactive : {0,1}, optional
  Wether to run the simulation in real-time, or offline (and replay it).

"""
import importlib.util
import os
import sys
import tempfile

from panda3d.core import load_prc_file_data

sys.path.insert(0, os.path.abspath(".."))
from core.scenario import (StateObserver, load_scenario_instance,  # noqa: E402
                           load_scene)
from gui.viewers import PhysicsViewer, Replayer  # noqa: E402

FPS = 500
DURATION = 4


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    if len(sys.argv) > 2:
        interactive = bool(sys.argv[2])
    else:
        interactive = False
    script = load_module("scene_script", path)
    scenario_data = script.DATA
    scene_data = scenario_data['scene']
    if interactive:
        scene = load_scene(scene_data, geom='LD', phys=True)
        app = PhysicsViewer(world=scene.world)
        scene.graph.reparent_to(app.models)
    else:
        dir_ = tempfile.mkdtemp()
        # Create the scene geometry.
        scene = load_scene(scene_data, geom='HD', phys=False)
        scene_path = os.path.join(dir_, "scene")
        scene.export_scene_to_egg(scene_path)
        # Run the instance.
        instance = load_scenario_instance(scenario_data, geom=None, phys=True)
        obs = StateObserver(instance.scene)
        print("Physically valid: ", instance.scene.check_physically_valid())
        instance.simulate(duration=DURATION, timestep=1/FPS, callbacks=[obs])
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
