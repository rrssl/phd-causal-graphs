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
import export
import spline2d as spl
from xp.config import t, w


def export_domino_run(filename, u, spline, sheetsize=(21, 29.7)):
    xy = np.column_stack(spl.splev(u, spline)) * 100
    a = spl.splang(u, spline, degrees=True)
    size = np.tile((t*100, w*100), (len(u), 1))

    extents = xy.ptp(axis=0) + w*100
    assert extents.size == 2
    if extents[1] < sheetsize[0] < extents[0] < sheetsize[1]:
        rot = np.array(((0, 1), (-1, 0)))
        xy = xy.dot(rot)
        a += 90
        extents = extents[::-1]
    xy = xy - (xy.min(axis=0) + xy.max(axis=0))/2 + np.asarray(sheetsize)/2

    cont = export.VectorFile(filename, sheetsize)
    cont.add_rectangles(xy, a, size)
    cont.save()


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
    export_domino_run(filename, u, spline)


if __name__ == "__main__":
    main()
