"""
Methods to generate the samples used to train the estimators. Each
n-dimensional sample represents the parameters of a domino distribution.

"""
from random import uniform

import numpy as np

from .config import t, w, h, TOPPLING_ANGLE
from .config import X_MIN, X_MAX, Y_MAX, A_MIN, A_MAX, MIN_SPACING, MAX_SPACING
from .domgeom import make_collision_box, has_contact, tilt_box_forward


def sample2D(n, filter_overlap=True, tilt_domino=True):
    """Sample the domino-pair parameter values with 2 DoFs (distance
    between centers and relative heading angle).

    The center of D2 lies on the line orthogonal to D1's largest face and
    going through D1's center.

    """
    samples = np.empty((n, 2))

    if filter_overlap:
        i = 0
        while i < n:
            d = uniform(X_MIN, X_MAX)
            a = uniform(0, A_MAX)  # Symmetry

            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (d, 0, h/2), (a, 0, 0))
            if tilt_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if has_contact(d1, d2):
                pass
            else:
                samples[i] = d, a
                i += 1
    else:
        samples[:, 0] = np.random.uniform(X_MIN, X_MAX, n)
        samples[:, 1] = np.random.uniform(0, A_MAX, n)  # Symmetry

    return samples


def sample3D(n, filter_overlap=True, tilt_domino=True):
    """Sample the domino-pair parameter values with 3 DoFs (relative position
    in the XY plane + relative heading angle).

    """
    samples = np.empty((n, 3))

    if filter_overlap:
        i = 0
        while i < n:
            x = uniform(X_MIN, X_MAX)
            y = uniform(0, Y_MAX)  # Symmetry
            a = uniform(A_MIN, A_MAX)

            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if has_contact(d1, d2):
                pass
            else:
                samples[i] = x, y, a
                i += 1
    else:
        samples[:, 0] = np.random.uniform(X_MIN, X_MAX, n)
        samples[:, 1] = np.random.uniform(0, Y_MAX, n)  # Symmetry
        samples[:, 2] = np.random.uniform(A_MIN, A_MAX, n)

    return samples


def sample4D(n, filter_overlap=True, tilt_domino=True):
    """Sample the domino-pair parameter values with 4 DoFs (relative position
    in the XY plane + relative heading angle + spacing between previous doms).

    """
    samples = np.empty((n, 4))

    if filter_overlap:
        i = 0
        while i < n:
            x = uniform(X_MIN, X_MAX)
            y = uniform(0, Y_MAX)  # Symmetry
            a = uniform(A_MIN, A_MAX)
            s = uniform(MIN_SPACING, MAX_SPACING)

            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if not has_contact(d1, d2):
                samples[i] = x, y, a, s
                i += 1
    else:
        samples[:, 0] = np.random.uniform(X_MIN, X_MAX, n)
        samples[:, 1] = np.random.uniform(0, Y_MAX, n)  # Symmetry
        samples[:, 2] = np.random.uniform(A_MIN, A_MAX, n)
        samples[:, 3] = np.random.uniform(MIN_SPACING, MAX_SPACING, n)

    return samples
