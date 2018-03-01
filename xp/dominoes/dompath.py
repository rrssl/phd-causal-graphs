"""
Utility functions related to domino paths.

"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(".."))
import spline2d as spl


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
