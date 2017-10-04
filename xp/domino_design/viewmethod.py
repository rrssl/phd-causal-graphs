"""
Run a method on an input sketch and view its output.

Parameters
----------
path : string
  Path to the sketch file.
mid : int
  Method id. See methods.py for the list of methods.

"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(".."))
from domino_design.config import t, w, PATH_SIZE_RATIO, SMOOTHING_FACTOR
from domino_design.methods import get_methods
from domino_design.viewdoms import show_dominoes


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    fname = sys.argv[1]
    mid = int(sys.argv[2])
    # Select method
    method = get_methods()[mid-1]
    print("Using method {}: {}".format(mid, method.__name__))
    # Load path
    path = np.load(fname)[0]
    # Translate, resize and smooth the path
    path -= path.min(axis=0)
    path *= PATH_SIZE_RATIO * math.sqrt(
            t * w / (path[:, 0].max() * path[:, 1].max()))
    spline = spl.get_smooth_path(path, s=SMOOTHING_FACTOR)

    u = method(spline)
    show_dominoes([u], [spline])


if __name__ == "__main__":
    main()
