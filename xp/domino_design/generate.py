"""
Generate a list of candidate splines from a list of input sketches.

Parameters
----------
nsplines : int
  Number of splines to generate.
skpaths : string or list of strings
  Paths to the input sketch(es).

"""
import math
import os
import pickle
import random
import sys

import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString
from shapely.geometry import Point

from config import MIN_SIZE_RATIO
from config import MIN_SMOOTHING_FACTOR
from config import MAX_SIZE_RATIO
from config import MAX_SMOOTHING_FACTOR
from config import t
from config import w
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


def test_valid_spline(spline):
    """Test if the domino path is geometrically valid.

    Supposing constant domino extents (t, w, h), a path is valid
    iff a circle of radius w, swept along the path, never
    intersects the path more than twice (once 'in', once 'out').

    """
    l = spl.arclength(spline)
    u = np.linspace(0, 1, int(l/t))
    vertices = np.column_stack(spl.splev(u, spline))
    path = LineString(vertices)
    base_circle = Point(0, 0).buffer(w)
    for x, y in vertices:
        circle = translate(base_circle, x, y)
        try:
            if len(circle.boundary.intersection(path)) > 2:
                return False
        except TypeError:  # Happens when intersection is a single point.
            pass
    return True


def generate_candidate_splines(sketches, size_rng, smoothing_rng, nsplines):
    splines = []
    # Randomly sample valid splines.
    while len(splines) < nsplines:
        sketch = random.choice(sketches)
        size_ratio = random.randint(*size_rng)
        smoothing_factor = random.uniform(*smoothing_rng)

        path = sketch[0]  # this will change when we accept several strokes
        # Translate, resize and smooth the path
        path -= path.min(axis=0)
        path *= size_ratio * math.sqrt(
                t * w / (path[:, 0].max() * path[:, 1].max()))
        spline = spl.get_smooth_path(path, s=smoothing_factor)
        if test_valid_spline(spline):
            splines.append(spline)
    return splines


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    nsplines = int(sys.argv[1])
    skpaths = sys.argv[2:]
    sketches = [np.load(skpath) for skpath in skpaths]

    splines = generate_candidate_splines(
            sketches,
            (MIN_SIZE_RATIO, MAX_SIZE_RATIO),
            (MIN_SMOOTHING_FACTOR, MAX_SMOOTHING_FACTOR),
            nsplines
            )

    with open("candidates.pkl", 'wb') as f:
        pickle.dump(splines, f)


if __name__ == "__main__":
    main()
