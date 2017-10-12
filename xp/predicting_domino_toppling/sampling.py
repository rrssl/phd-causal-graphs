"""
Sampling the parameter space.

Parameters
----------
nsam : int
  Number of samples.
ndof : int, optional
  Number of degrees of freedom. Defaults to 3 (relative x, y, and angle).

"""
import os
from random import uniform
import sys

import numpy as np

from config import t, w, h, TOPPLING_ANGLE
from config import X_MIN, X_MAX, Y_MIN, Y_MAX, A_MIN, A_MAX
sys.path.insert(0, os.path.abspath("../.."))
from xp.domgeom import make_box, tilt_box_forward, has_contact


def sample2d(n, bounds, filter_overlap=True):
    """Sample the domino-pair parameter values with 2 DoFs (distance
    between centers and relative heading angle).

    The center of D2 is supposed to lie on the line orthogonal to D1's largest
    face and going through D1's center.

    """
    samples = np.empty((n, 2))

    if filter_overlap:
        i = 0
        while i < n:
            d = uniform(*bounds[0])  # Distance between centers
            a = uniform(*bounds[1])  # Heading angle between D1 and D2

            d1 = make_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_box((t, w, h), (d, 0, h/2), (a, 0, 0))
            tilt_box_forward(d1, TOPPLING_ANGLE + 1)
            if has_contact(d1, d2):
                pass
            else:
                samples[i] = d, a
                i += 1
    else:
        samples[:, 0] = np.random.uniform(bounds[0][0], bounds[0][1], n)
        samples[:, 1] = np.random.uniform(bounds[1][0], bounds[1][1], n)

    return samples


def sample3d(n, bounds, filter_overlap=True):
    """Sample the domino-pair parameter values with 3 DoFs (relative position
    in the XY plane + relative heading angle).

    """
    samples = np.empty((n, 3))

    if filter_overlap:
        i = 0
        while i < n:
            x = uniform(*bounds[0])
            y = uniform(*bounds[1])
            a = uniform(*bounds[2])

            d1 = make_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_box((t, w, h), (x, y, h/2), (a, 0, 0))
            tilt_box_forward(d1, TOPPLING_ANGLE + 1)
            if has_contact(d1, d2):
                pass
            else:
                samples[i] = x, y, a
                i += 1
    else:
        samples[:, 0] = np.random.uniform(bounds[0][0], bounds[0][1], n)
        samples[:, 1] = np.random.uniform(bounds[1][0], bounds[1][1], n)
        samples[:, 2] = np.random.uniform(bounds[2][0], bounds[2][1], n)

    return samples


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    nsam = int(sys.argv[1])
    ndim = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if ndim == 2:
        # Angle is > 0 because of symmetry.
        bounds = ((X_MIN, X_MAX), (0, A_MAX))
        samples = sample2d(nsam, bounds)
    elif ndim == 3:
        # Y is > 0 because of symmetry.
        bounds = ((X_MIN, X_MAX), (0, Y_MAX), (A_MIN, A_MAX))
        samples = sample3d(nsam, bounds)
    else:
        print("No method implemented for this number of dimensions.")
        return

    name = "samples-{}D.npy".format(ndim)
    np.save(name, samples)


if __name__ == "__main__":
    main()
