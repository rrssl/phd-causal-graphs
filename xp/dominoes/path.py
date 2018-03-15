"""
Utility functions related to domino paths.

"""
import numpy as np
from scipy.signal import argrelmax
from shapely.affinity import rotate, translate
from shapely.geometry import box, LineString, Point

import core.spline2d as spl


def get_rel_coords(u, spline):
    """Get the relative configuration of each domino wrt the previous.

    The position and angle are reflected so that Y is always positive.
    The relative orientation is in [-180, 180).

    """
    # Get local Cartesian coordinates
    # Change origin
    xi, yi = spl.splev(u, spline)
    xi = np.diff(xi)
    yi = np.diff(yi)
    # Rotate by -a_i-1
    ai = spl.splang(u, spline, degrees=False)
    ci_ = np.cos(ai[:-1])
    si_ = np.sin(ai[:-1])
    xi_r = xi*ci_ + yi*si_
    yi_r = -xi*si_ + yi*ci_
    # Get relative angles
    ai_r = np.degrees(np.diff(ai))
    ai_r = (ai_r + 180) % 360 - 180  # Convert from [0, 360) to [-180, 180)
    # Symmetrize
    ai_r = np.copysign(ai_r, yi_r)
    yi_r = np.abs(yi_r)

    return np.column_stack((xi_r, yi_r, ai_r))


class DominoSequence:
    """More generic than a domino path, it simply stores the coordinates
    of a sequence of dominoes.

    """
    def __init__(self, coords):
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, key):
        return self.coords[key]


class DominoPath(DominoSequence):
    """Sequence of dominoes, with the assumption that the sequence follows
    a spline path.

    """
    def __init__(self, u, spline):
        self.u = u
        self.spline = spline
        coords = np.zeros((len(u), 3))
        coords[:, 0], coords[:, 1] = spl.splev(u, spline)
        coords[:, 2] = spl.splang(u, spline)
        super().__init__(coords)


class DominoPathChecker:
    """Checks the feasibility of a spline as a domino path.

    All the tests suppose constant domino extents (t, w, h).

    """
    def __init__(self, spline, domino_extents):
        self.spline = spline
        self.t, self.w, self.h = domino_extents
        length = spl.arclength(spline)
        # 'u' here is a dense sampling of the spline. Not the same as
        # 'u' in DominoPath.
        self.u = np.linspace(0, 1, int(length/self.t))
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
        base_circle = Point(0, 0).buffer(self.w)
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
        base_rect = box(0, -self.w/2, self.h, self.w/2)
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
        maxima = maxima[curvature[maxima] > 2/self.w]
        if maxima.size == 0:
            return False
        base_rect = box(-self.t/2, -self.w/2, self.t/2, self.w/2)
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
            #  fig, ax = plt.subplots()
            #  ax.plot(*self.vertices.T)
            #  ax.add_patch(PolygonPatch(rect_at_max, alpha=.5, zorder=2))
            #  ax.add_patch(PolygonPatch(rect_before, alpha=.5, zorder=2))
            #  ax.add_patch(PolygonPatch(rect_after, alpha=.5, zorder=2))
            #  ax.set_aspect('equal', 'datalim')
            #  ax.scatter(*self.vertices[[pid_before, maxid, pid_after]].T)
            #  plt.show()

            if (rect_before.intersects(rect_at_max)
                    or rect_at_max.intersects(rect_after)
                    or rect_after.intersects(rect_before)):
                return True
        return False
