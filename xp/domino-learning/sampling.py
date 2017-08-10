"""
Sampling the parameter space. Enter n as the number of samples.

Domino dimensions are specified in config.py.
Sampling-specific options (sampled parameters, sampling bounds, file name) are
defined in the main().

"""
from math import pi, atan
import numpy as np
from random import uniform
import sys

from config import t, w, h
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

        d1 = make_box((t, w, h), (0, 0, h*.5), (0, 0, 0))
        d2 = make_box((t, w, h), (d, 0, h*.5), (a, 0, 0))
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

        d1 = make_box((t, w, h), (0, 0, h*.5), (0, 0, 0))
        d2 = make_box((t, w, h), (x, y, h*.5), (a, 0, 0))
        tilt_box_forward(d1, atan(t / h) * 180 / pi + 1.)
        if has_contact(d1, d2):
            pass
        else:
            samples[i] = x, y, a
            i += 1

    return samples


def main():
    if len(sys.argv) <= 1:
        print("Please enter the number of samples.")
        return
    nsam = int(sys.argv[1])

    # TODO: make them optional script arguments
    ndim = 3
    name = "samples-{}D.npy".format(ndim)

    if ndim == 2:
        # Angle is > 0 because of symmetry.
        bounds = ((t, 1.5*h), (0, 90))
        sample = sample2d
    else:
        # Y is > 0 because of symmetry.
        bounds = ((t, 1.5*h), (0, w), (-90, 90))
        sample = sample3d
    s = sample(nsam, bounds)
    np.save(name, s)


if __name__ == "__main__":
    main()
