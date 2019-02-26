import os
import sys

from panda3d.core import load_prc_file_data

sys.path.insert(0, os.path.abspath(".."))
from gui.viewers import CausalGraphEditor  # noqa: E402


def main():
    load_prc_file_data("", "win-origin 500 200")
    app = CausalGraphEditor()
    app.run()


if __name__ == "__main__":
    main()
