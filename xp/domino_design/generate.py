"""
Generate a list of candidate paths from a list of input sketches.

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


def test_valid_path(spline):
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


def generate_candidate_paths(sketches, size_rng, smoothing_rng, npaths):
    paths = []
    # Randomly sample valid paths.
    while len(paths) < npaths:
        sketch = random.choice(sketches)
        size_ratio = random.randint(*size_rng)
        smoothing_factor = random.uniform(*smoothing_rng)

        path = np.load(sketch)[0]
        # Translate, resize and smooth the path
        path -= path.min(axis=0)
        path *= size_ratio * math.sqrt(
                t * w / (path[:, 0].max() * path[:, 1].max()))
        spline = spl.get_smooth_path(path, s=smoothing_factor)
        if test_valid_path(spline):
            paths.append(spline)
    return paths


def main():
    if len(sys.argv) <= 1:
        print("Please provide a number of candidates and at least one path.")
        return
    npaths = int(sys.argv[1])
    sketches = sys.argv[2:]
    paths = generate_candidate_paths(
            sketches,
            (MIN_SIZE_RATIO, MAX_SIZE_RATIO),
            (MIN_SMOOTHING_FACTOR, MAX_SMOOTHING_FACTOR),
            npaths
            )
    with open("candidates.pkl", 'wb') as f:
        pickle.dump(paths, f)


if __name__ == "__main__":
    main()
