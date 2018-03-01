"""
Run a method on a spline and view its output.

Parameters
----------
spath : string
  Path to the splines.
sid : int
  Spline id.
mid : int
  Method id. See methods.py for the list of methods.

"""
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl
from xp.dominoes.creation import get_methods
from xp.viewdoms import show_dominoes


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])
    mid = int(sys.argv[3])
    # Select method
    method = get_methods()[mid-1]
    # Load spline
    with open(spath, 'rb') as f:
        splines = pickle.load(f)
    spline = splines[sid]


    u = method(spline)
    show_dominoes([u], [spline])


if __name__ == "__main__":
    main()
