import json
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from core.scenario import (StateObserver, load_scenario,  # noqa:E402
                           simulate_scene)


FPS = 240
DURATION = 2


def main():
    if len(sys.argv) < 3:
        return
    json_path = sys.argv[1]
    out_dir = sys.argv[2]
    with open(json_path, 'r') as f:
        scenario_data = json.load(f)
    scenario = load_scenario(scenario_data, geom='HD', phys=False)
    scenario.scene.export_scene_to_egg(out_dir + "scene.egg")
    scenario = load_scenario(scenario_data, geom=None, phys=True)
    obs = StateObserver(scenario.scene)
    simulate_scene(scenario.scene, duration=DURATION, timestep=1/FPS,
                   callbacks=[obs])
    obs.export(out_dir + "simu.pkl", fps=FPS)


if __name__ == "__main__":
    main()
