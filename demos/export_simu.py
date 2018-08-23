import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from core.scenario import load, simulate_scene, StateObserver  # noqa:E402

FPS = 240


def main():
    scenario = load("../scenarios/simple.json", geom='HD')
    out_dir = "data/export_simu/"
    scenario.export_scene_to_egg(out_dir + "scene.egg")
    obs = StateObserver(scenario.scene)
    simulate_scene(scenario.scene, duration=2., timestep=1/FPS,
                   callbacks=[obs])
    obs.export(out_dir + "simu.pkl", fps=FPS)


if __name__ == "__main__":
    main()
