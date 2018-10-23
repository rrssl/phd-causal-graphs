import json
import os
import sys
import tempfile

from panda3d.core import load_prc_file_data

sys.path.insert(0, os.path.abspath(".."))
from core.scenario import (StateObserver, load_scenario_instance,  # noqa:E402
                           load_scene)
from gui.viewers import Replayer  # noqa: E402


FPS = 500
DURATION = 2


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    with open(path, 'r') as f:
        scenario_data = json.load(f)
    dir_ = tempfile.mkdtemp()
    # Create the scene geometry.
    scene_data = scenario_data['scene']
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
    app.run()


if __name__ == "__main__":
    main()
