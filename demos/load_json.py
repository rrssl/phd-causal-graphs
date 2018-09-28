import json
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from core.scenario import load_scenario_instance  # noqa:E402
from gui.viewers import PhysicsViewer  # noqa:E402


def main():
    with open("../scenarios/simple.json", 'r') as f:
        scenario_data = json.load(f)
    scenario = load_scenario_instance(scenario_data, geom='HD')
    app = PhysicsViewer(world=scenario.scene.world)
    app.cam_distance = 1
    app.min_cam_distance = .01
    app.camLens.set_near(.01)
    app.zoom_speed = .01
    scenario.scene.graph.reparent_to(app.models)
    app.run()


if __name__ == "__main__":
    main()
