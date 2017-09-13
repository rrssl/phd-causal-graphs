"""
Sampling the parameter space.

The domino dimensions and sampling space bounds+dimensions are specified in
'config.py'.

Parameters
----------
n : int
  Number of samples.

"""
from math import pi, atan
import numpy as np
from random import uniform
import sys

from config import t, w, h
from config import X_MIN, X_MAX, Y_MIN, Y_MAX, A_MIN, A_MAX, SAMPLING_NDIM
from functions import make_box, tilt_box_forward, has_contact


def sample2d(n, bounds):
    """Sample the domino-pair parameter values with 2 DoFs (distance
    between centers and relative heading angle).

    The center of D2 is supposed to lie on the line orthogonal to D1's largest
    face and going through D1's center.

    """
    samples = np.empty((n, 2))
    i = 0
    while i < n:
        d = uniform(*bounds[0])  # Distance between centers
        a = uniform(*bounds[1])  # Heading angle between D1 and D2

        d1 = make_box((t, w, h), (0, 0, h/2), (0, 0, 0))
        d2 = make_box((t, w, h), (d, 0, h/2), (a, 0, 0))
        tilt_box_forward(d1, atan(t / h) * 180 / pi + 1.)
        if has_contact(d1, d2):
            pass
        else:
            samples[i] = d, a
            i += 1

    return samples


def sample3d(n, bounds):
    """Sample the domino-pair parameter values with 3 DoFs (relative position
    in the XY plane + relative heading angle).

    """
    samples = np.empty((n, 3))
    i = 0
    while i < n:
        x = uniform(*bounds[0])
        y = uniform(*bounds[1])
        a = uniform(*bounds[2])

        d1 = make_box((t, w, h), (0, 0, h/2), (0, 0, 0))
        d2 = make_box((t, w, h), (x, y, h/2), (a, 0, 0))
        tilt_box_forward(d1, atan(t / h) * 180 / pi + 1.)
        if has_contact(d1, d2):
            pass
        else:
            samples[i] = x, y, a
            i += 1

    return samples


def main():
    if len(sys.argv) <= 1:
        print(__doc__)
        return
    nsam = int(sys.argv[1])

    if SAMPLING_NDIM == 2:
        # Angle is > 0 because of symmetry.
        bounds = ((X_MIN, X_MAX), (0, A_MAX))
        samples = sample2d(nsam, bounds)
        samples_sym = [[s[0], -s[1]] for s in samples if s[1]]
        samples = np.vstack((samples, samples_sym))
    elif SAMPLING_NDIM == 3:
        # Y is > 0 because of symmetry.
        bounds = ((X_MIN, X_MAX), (0, Y_MAX), (A_MIN, A_MAX))
        samples = sample3d(nsam, bounds)
        samples_sym = [[s[0], -s[1], s[2]] for s in samples if s[1]]
        samples = np.vstack((samples, samples_sym))
    else:
        print("No method implemented for this number of dimensions.")
        return

    name = "samples-{}D.npy".format(SAMPLING_NDIM)
    np.save(name, samples)


if __name__ == "__main__":
    main()
