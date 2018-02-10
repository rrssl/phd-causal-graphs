"""
Fit a spline to the input polyline.

Parameters
----------
path : string
  Path to the sketch file.
smoothing : float
  Smoothing factor.
width : float
  The width (in cm) to resize this sketch to.

"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl  # noqa


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    path = sys.argv[1]
    smoothing = float(sys.argv[2])
    width = float(sys.argv[3]) / 100
    # Load path
    sketch = np.load(path)[0]
    # Translate, resize and smooth the path
    sketch -= sketch.min(axis=0)
    sketch *= width / sketch[:, 0].max()
    spline = spl.get_smooth_path(sketch, s=smoothing)
    # Show
    fig, ax = plt.subplots()
    ax.plot(*spl.splev(np.linspace(0, 1, 100), spline))
    ax.set_aspect('equal', 'datalim')
    plt.show()
    # Save
    basename = os.path.splitext(path)[0]
    with open(basename + ".pkl", 'bw') as f:
        pickle.dump([spline], f)


if __name__ == "__main__":
    main()
