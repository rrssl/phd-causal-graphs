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
from scipy.signal import argrelmax
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import Point

from config import MIN_SIZE_RATIO
from config import MIN_SMOOTHING_FACTOR
from config import MAX_SIZE_RATIO
from config import MAX_SMOOTHING_FACTOR
from config import t, w, h
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


class DominoPathTester:
    """Checking for infeasible domino paths.

    All the tests suppose constant domino extents (t, w, h).
    """
    def __init__(self, spline):
        self.spline = spline
        length = spl.arclength(spline)
        self.u = np.linspace(0, 1, int(length/t))
        self.angles = spl.splang(self.u, spline)
        self.vertices = np.column_stack(spl.splev(self.u, spline))
        self.path = LineString(self.vertices)

    def check(self):
        """Run all tests."""
        return (not self.has_nonlocal_overlaps()
                and not self.has_toppling_interferences()
                and not self.has_overly_acute_turns())

    def has_nonlocal_overlaps(self):
        """Detect the possibility of nonlocal domino overlaps (including
        self-intersections of the path), i.e. dominoes overlapping with
        other dominoes further down the path.

        Ensure that a circle of radius w, swept along the path, never
        intersects the remaining path more than twice (once 'in', once 'out').

        """
        if not self.path.is_simple:
            return True
        base_circle = Point(0, 0).buffer(w)
        rem_path = LineString(self.path)
        for x, y in self.vertices[:-2]:
            rem_path = LineString(rem_path.coords[1:])
            circle = translate(base_circle, x, y)
            try:
                if len(circle.boundary.intersection(rem_path)) > 2:
                    return True
            except TypeError:  # Happens when intersection is a single point.
                pass
        return False

    def has_toppling_interferences(self):
        """Detect the possibility of dominoes falling on dominoes that are
        not immediately after them.

        Ensure that a w-by-h box, swept along the path (with the middle
        of the small side as origin), and aligned with the tangent,
        never intersects the remainder of the path more than twice
        (once 'in', once 'out').

        """
        base_rect = box(0, -w/2, h, w/2)
        rem_path = LineString(self.path)
        for xy, ang in zip(self.vertices[:-2], self.angles[:-2]):
            rem_path = LineString(rem_path.coords[1:])
            rect = translate(rotate(base_rect, ang), *xy)
            try:
                if len(rect.boundary.intersection(rem_path)) > 2:
                    return True
            except TypeError:  # Happens when intersection is a single point.
                pass
        return False

    def has_overly_acute_turns(self):
        """Detect the presence of turns that are too narrow to safely place
        dominoes.

        For each point P of local maximal curvature with a radius smaller
        than w/2:
         -- Try to find the two possible curve samples P- and P+,
         respectively before and after that point, such that their tangent
         makes a 45 degree angle with the tangent at P.
         -- Put a t-by-w rectangle at each of these points.
        Ensure that none of these rectangles overlap.

        Simply defining an upper bound on curvature would be too coarse,
        as high-curvature points can work very well as long as
        the angle of the 'arms' before and after is wide enough.

        The result will only be meaningful if has_nonlocal_overlaps() = False.

        """
        curvature = spl.curvature(self.spline, self.u)
        maxima = argrelmax(curvature)[0]
        maxima = maxima[curvature[maxima] > 2/w]
        if maxima.size == 0:
            return False
        base_rect = box(-t/2, -w/2, t/2, w/2)
        for maxid in maxima:
            angle_at_max = self.angles[maxid]
            # Find the last vertex id going BACKWARDS from the maximum
            # such that the tangential angle with it is < 45 degrees.
            try:
                pid_before = next(
                        i for i in range(maxid-1, -1, -1)
                        if abs(self.angles[i] - angle_at_max) >= 45)
            except StopIteration:
                # Such point was not found, no need to continue the check
                continue
            # Find the last vertex id going FORWARDS from the maximum
            # such that the tangential angle with it is < 45 degrees.
            try:
                pid_after = next(
                        i for i in range(maxid+1, len(self.angles))
                        if abs(self.angles[i] - angle_at_max) >= 45)
            except StopIteration:
                # Such point was not found, no need to continue the check
                continue

            # Evaluate intersection between rectangles
            rect_at_max = translate(
                    rotate(base_rect, angle_at_max),
                    *self.vertices[maxid])
            rect_before = translate(
                    rotate(base_rect, self.angles[pid_before]),
                    *self.vertices[pid_before])
            rect_after = translate(
                    rotate(base_rect, self.angles[pid_after]),
                    *self.vertices[pid_after])

            #  import matplotlib.pyplot as plt
            #  from descartes import PolygonPatch
            #  plt.plot(*self.vertices.T)
            #  plt.gca().add_patch(PolygonPatch(rect_at_max, alpha=.5, zorder=2))
            #  plt.gca().add_patch(PolygonPatch(rect_before, alpha=.5, zorder=2))
            #  plt.gca().add_patch(PolygonPatch(rect_after, alpha=.5, zorder=2))
            #  plt.gca().set_aspect('equal', 'datalim')
            #  plt.scatter(*self.vertices[[pid_before, maxid, pid_after]].T)
            #  plt.ioff()
            #  plt.show()

            if (rect_before.intersects(rect_at_max)
                    or rect_at_max.intersects(rect_after)
                    or rect_after.intersects(rect_before)):
                return True
        return False


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
        tester = DominoPathTester(spline)
        if tester.check():
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
