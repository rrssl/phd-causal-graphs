"""
Methods to generate the samples used to train the estimators. Each
n-dimensional sample represents the parameters of a domino distribution.

Scenarios
1. 2 dominoes; the second one is orthogonal to the line joining the centers.
    (2 DoFs: distance between centers and polar angle of the second center.)
2. 2 dominoes; the second one is free.
    (3 DoFs: center coordinates and relative angle of the second domino.)
3. n dominoes; the n-1 first are equidistant on a straight line and the last
    one is free.
    (4 DoFs: 3 from scenario (2) + distance between the n-1 first dominoes.)
"""
from enum import Enum
import math

import chaospy as cp
import numpy as np

from .config import t, w, h, TOPPLING_ANGLE
from .config import X_MIN, X_MAX, Y_MAX, A_MIN, A_MAX, MIN_SPACING, MAX_SPACING
from .domgeom import make_collision_box, has_contact, tilt_box_forward


class Scenario(Enum):
    TWO_DOMS_LAST_RADIAL = 1,
    TWO_DOMS_LAST_FREE = 2
    N_DOMS_STRAIGHT_LAST_FREE = 3
    THREE_DOMS_TWO_LAST_FREE = 4


def sample(n, scenario, generator_rule='H', filter_rules=None):
    """Generate training samples describing domino distributions.

    Parameters
    ----------
    n : int
      Number of samples.
    scenario : Scenario
      Domino run scenario.
    generator_rule : string, optional
      'rule' parameter used by chaospy's sampler.
    filter_rules : dict, optional
      Additional rules given to the scenario (e.g. 'filter_overlap,
      'tilt_first_domino').

    """
    cp.seed(n)
    if scenario is Scenario.TWO_DOMS_LAST_RADIAL:
        return sample_2_doms_last_radial(n, **filter_rules)
    elif scenario is Scenario.TWO_DOMS_LAST_FREE:
        return sample_2_doms_last_free(n, **filter_rules)
    elif scenario is Scenario.N_DOMS_STRAIGHT_LAST_FREE:
        return sample_n_doms_straight_last_free(n, **filter_rules)
    elif scenario is Scenario.THREE_DOMS_TWO_LAST_FREE:
        return sample_3_doms_2_last_free(n, **filter_rules)


def sample_2_doms_last_radial(
        n, rule='H', filter_overlap=True, tilt_first_domino=True):
    """Sample the domino-pair parameter values with 2 DoFs (distance
    between centers and relative heading angle).

    The center of D2 lies on the line orthogonal to D1's largest face and
    going through D1's center.

    """
    dist = cp.J(cp.Uniform(X_MIN, X_MAX), cp.Uniform(0, A_MAX))
    cos = math.cos
    sin = math.sin
    pi = math.pi

    if filter_overlap:
        samples = np.empty((n, 2))
        i = 0
        while i < n:
            r, a = dist.sample(1, rule=rule).reshape(2)
            a_ = a * pi / 180
            x = r * cos(a_)
            y = r * sin(a_)
            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_first_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if has_contact(d1, d2):
                pass
            else:
                samples[i] = r, a
                i += 1
    else:
        samples = dist.sample(n, rule=rule).T

    return samples


def sample_2_doms_last_free(
        n, rule='H', filter_overlap=True, tilt_first_domino=True):
    """Sample the domino-pair parameter values with 3 DoFs (relative position
    in the XY plane + relative heading angle).

    """
    dist = cp.J(cp.Uniform(X_MIN, X_MAX),
                cp.Uniform(0, Y_MAX),  # Symmetry
                cp.Uniform(A_MIN, A_MAX))

    if filter_overlap:
        samples = np.empty((n, 3))
        i = 0
        while i < n:
            x, y, a = dist.sample(1, rule=rule).reshape(3)
            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_first_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if has_contact(d1, d2):
                pass
            else:
                samples[i] = x, y, a
                i += 1
    else:
        samples = dist.sample(n, rule=rule).T

    return samples


def sample_n_doms_straight_last_free(
        n, rule='H', filter_overlap=True, tilt_first_domino=True):
    """Sample the domino-pair parameter values with 4 DoFs (relative position
    in the XY plane + relative heading angle + spacing between previous doms).

    """
    dist = cp.J(cp.Uniform(X_MIN, X_MAX),
                cp.Uniform(0, Y_MAX),  # Symmetry
                cp.Uniform(A_MIN, A_MAX),
                cp.Uniform(MIN_SPACING, MAX_SPACING))

    if filter_overlap:
        samples = np.empty((n, 4))
        i = 0
        while i < n:
            x, y, a, s = dist.sample(1, rule=rule).reshape(4)
            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_first_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if not has_contact(d1, d2):
                samples[i] = x, y, a, s
                i += 1
    else:
        samples = dist.sample(n, rule=rule).T

    return samples


def sample_3_doms_2_last_free(
        n, rule='H', filter_overlap=True, tilt_first_domino=True):
    """Sample a domino-triplet parameter values with 6 DoFs (relative
    transforms of domino 2 vs 1 and 3 vs 2.)

    """
    n_pair = math.ceil(math.sqrt(n))
    samples_pair = sample_2_doms_last_free(
            n_pair, rule, filter_overlap, tilt_first_domino)
    samples = np.concatenate(
            (np.repeat(samples_pair, n_pair, axis=0),
             np.tile(samples_pair, (n_pair, 1))),
            axis=1)[:n]
    return samples
