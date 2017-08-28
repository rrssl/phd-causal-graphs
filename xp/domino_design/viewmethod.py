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

from config import t, w, PATH_SIZE_RATIO, SMOOTHING_FACTOR
from methods import get_methods
from viewdoms import show_dominoes

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    fname = sys.argv[1]
    mid = int(sys.argv[2])
    # Select method
    method = get_methods()[mid-1]
    # Load path
    path = np.load(fname)[0]
    # Translate, resize and smooth the path
    path -= path.min(axis=0)
    path *= PATH_SIZE_RATIO * math.sqrt(
            t * w / (path[:, 0].max() * path[:, 1].max()))
    spline = spl.get_smooth_path(path, s=SMOOTHING_FACTOR)

    u = method(spline)
    show_dominoes(u, spline)


if __name__ == "__main__":
    main()