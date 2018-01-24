"""
Export the domino distribution to PDF.

Parameters
----------
spath : string
  Path to the .pkl file of splines.
dpath : string
  Path to the .npz file of distributions.
did : int
  Index of the domino distribution in the file.

"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
import export  # noqa
import spline2d as spl  # noqa
from xp.config import t, w  # noqa


def export_domino_run(filename, coords, sheetsize=(21, 29.7)):
    xy = coords[:, :2] * 100
    a = coords[:, 2]
    size = np.tile((t*100, w*100), (coords.shape[0], 1))

    extents = xy.ptp(axis=0) + w*100
    if extents[1] < sheetsize[0] < extents[0] < sheetsize[1]:
        xy[:, [0, 1]] = xy[:, [1, 0]]
        xy[:, 0] *= -1
        a += 90
    xy = xy - (xy.min(axis=0) + xy.max(axis=0))/2 + np.asarray(sheetsize)/2

    vec = export.VectorFile(filename, sheetsize)
    vec.add_rectangles(xy, a, size)
    vec.add_text("up", (2, 2))
    vec.save()


def export_domino_run_from_path(filename, u, spline, sheetsize=(21, 29.7)):
    x, y = spl.splev(u, spline)
    a = spl.splang(u, spline, degrees=True)
    export_domino_run(filename, np.column_stack((x, y, a)), sheetsize)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    spath = sys.argv[1]
    dpath = sys.argv[2]
    sid = int(sys.argv[3])

    with open(spath, 'rb') as f:
        spline = pickle.load(f)[sid]
    u = np.load(dpath)['arr_{}'.format(sid)]

    filename = os.path.splitext(dpath)[0]
    export_domino_run_from_path(filename, u, spline)


if __name__ == "__main__":
    main()
