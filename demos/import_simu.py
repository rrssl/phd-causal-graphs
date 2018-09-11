import os
import sys

from panda3d.core import load_prc_file_data

sys.path.insert(0, os.path.abspath(".."))
from gui.viewers import Replayer  # noqa: E402


def main():
    if len(sys.argv) < 3:
        return
    scene_path = sys.argv[1]
    simu_path = sys.argv[2]
    load_prc_file_data("", "win-origin 500 200")
    app = Replayer(scene_path, simu_path)
    app.cam_distance = 1
    app.min_cam_distance = .01
    app.camLens.set_near(.01)
    app.zoom_speed = .01
    app.run()


if __name__ == "__main__":
    main()
