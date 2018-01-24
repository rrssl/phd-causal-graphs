"""
Parameters
----------
spath : string
  Path to the .pkl file of splines.
sid : int
  Index of the spline in the file.

"""
import os
import sys
import pickle

import numpy as np
from scipy.interpolate import splev

sys.path.insert(0, os.path.abspath("../.."))
import export  # noqa


def export_spline(filename, spline, sheetsize=(21, 29.7)):
    xy = np.column_stack(splev(np.linspace(0, 1, 1000), spline)) * 100

    extents = xy.ptp(axis=0)
    if extents[1] < sheetsize[0] < extents[0] < sheetsize[1]:
        xy[:, [0, 1]] = xy[:, [1, 0]]
        xy[:, 0] *= -1
    xy = xy - (xy.min(axis=0) + xy.max(axis=0))/2 + np.asarray(sheetsize)/2

    vec = export.VectorFile(filename, sheetsize)
    vec.add_polyline(xy)
    vec.add_text("up", (2, 2))
    vec.save()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    sid = int(sys.argv[2])
    with open(spath, 'rb') as f:
        spline = pickle.load(f)[sid]

    filename = os.path.splitext(spath)[0]
    export_spline(filename, spline)


if __name__ == "__main__":
    main()
