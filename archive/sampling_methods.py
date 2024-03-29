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
4. 3 dominoes; the 2 last are free.
    (6 DoFs: 2x3 deom scenario (2).)

"""
from enum import Enum
import math

import chaospy as cp
import numpy as np

from .config import t, w, h, TOPPLING_ANGLE
from .config import (
        X_MIN, X_MAX, Y_MIN, Y_MAX, A_MIN, A_MAX, MIN_SPACING, MAX_SPACING)
from .domgeom import make_collision_box, has_contact, tilt_box_forward


class Scenario(Enum):
    TWO_DOMS_LAST_RADIAL = 1
    TWO_DOMS_LAST_FREE = 2
    P_DOMS_STRAIGHT_LAST_FREE = 3
    THREE_DOMS_TWO_LAST_FREE = 4
    P_DOMS_STRAIGHT_TWO_LAST_FREE = 5


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
    if generator_rule == 'R':
        cp.seed(n)

    if scenario is Scenario.TWO_DOMS_LAST_RADIAL:
        return sample_2_doms_last_radial(n, generator_rule, **filter_rules)
    elif scenario is Scenario.TWO_DOMS_LAST_FREE:
        return sample_2_doms_last_free(n, generator_rule, **filter_rules)
    elif scenario is Scenario.P_DOMS_STRAIGHT_LAST_FREE:
        return sample_p_doms_straight_last_free(
                n, generator_rule, **filter_rules)
    elif scenario is Scenario.THREE_DOMS_TWO_LAST_FREE:
        return sample_3_doms_2_last_free(n, generator_rule, **filter_rules)
    elif scenario is Scenario.P_DOMS_STRAIGHT_TWO_LAST_FREE:
        return sample_p_doms_straight_2_last_free(
                n, generator_rule, **filter_rules)


def sample_2_doms_last_radial(
        n, rule='H', filter_overlap=True, tilt_first_domino=True,
        reduce_symmetry=True):
    """Sample the domino-pair parameter values with 2 DoFs (distance
    between centers and relative heading angle).

    The center of D2 lies on the line orthogonal to D1's largest face and
    going through D1's center.

    """
    max_trials = 2 * n
    dist = cp.J(
            cp.Uniform(X_MIN, X_MAX),
            cp.Uniform((0 if reduce_symmetry else A_MIN), A_MAX))
    cos = math.cos
    sin = math.sin
    pi = math.pi

    if filter_overlap:
        cand_samples = dist.sample(max_trials, rule=rule)
        samples = np.empty((n, 2))
        n_valid = 0
        for r, a in cand_samples.T:
            a_ = a * pi / 180
            x = r * cos(a_)
            y = r * sin(a_)
            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_first_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if not has_contact(d1, d2):
                samples[n_valid] = r, a
                n_valid += 1
                if n_valid == n:
                    break
        else:
            print("Ran out of trials")
    else:
        samples = dist.sample(n, rule=rule).T

    return samples


def sample_2_doms_last_free(
        n, rule='H', filter_overlap=True, tilt_first_domino=True,
        reduce_symmetry=True):
    """Sample the domino-pair parameter values with 3 DoFs (relative position
    in the XY plane + relative heading angle).

    """
    max_trials = 2 * n
    dist = cp.J(cp.Normal((X_MIN+X_MAX)/2, (X_MAX-X_MIN)/4),
                cp.Normal((Y_MIN+Y_MAX)/2, (Y_MAX-Y_MIN)/4),
                cp.Normal((A_MIN+A_MAX)/2, (A_MAX-A_MIN)/4))
    #  dist = cp.J(cp.Uniform(X_MIN, X_MAX),
    #              cp.Uniform((0 if reduce_symmetry else Y_MIN), Y_MAX),
    #              cp.Uniform(A_MIN, A_MAX))

    if filter_overlap:
        cand_samples = dist.sample(max_trials, rule=rule)
        if reduce_symmetry:
            cand_samples[1] = abs(cand_samples[1])
        samples = np.empty((n, 3))
        n_valid = 0
        for x, y, a in cand_samples.T:
            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_first_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if not has_contact(d1, d2):
                samples[n_valid] = x, y, a
                n_valid += 1
                if n_valid == n:
                    break
        else:
            print("Ran out of trials")
    else:
        samples = dist.sample(n, rule=rule).T

    return samples


def sample_p_doms_straight_last_free(
        n, rule='H', filter_overlap=True, tilt_first_domino=True,
        reduce_symmetry=True):
    """Sample the domino-pair parameter values with 4 DoFs (relative position
    in the XY plane + relative heading angle + spacing between previous doms).

    """
    max_trials = 2 * n
    dist = cp.J(cp.Uniform(X_MIN, X_MAX),
                cp.Uniform((0 if reduce_symmetry else Y_MIN), Y_MAX),
                cp.Uniform(A_MIN, A_MAX),
                cp.Uniform(MIN_SPACING, MAX_SPACING))

    if filter_overlap:
        cand_samples = dist.sample(max_trials, rule=rule)
        samples = np.empty((n, 4))
        n_valid = 0
        for x, y, a, s in cand_samples.T:
            d1 = make_collision_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_collision_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if tilt_first_domino:
                tilt_box_forward(d1, TOPPLING_ANGLE+1)
            if not has_contact(d1, d2):
                samples[n_valid] = x, y, a, s
                n_valid += 1
                if n_valid == n:
                    break
        else:
            print("Ran out of trials")
    else:
        samples = dist.sample(n, rule=rule).T

    return samples


def sample_3_doms_2_last_free(
        n, rule='H', filter_overlap=True, tilt_first_domino=True,
        reduce_symmetry=True):
    """Sample domino-triplet parameter values with 6 DoFs (relative transforms
    of domino 2 vs 1 and 3 vs 2.)

    """
    samples_1 = sample_2_doms_last_free(
            n, rule, filter_overlap, tilt_first_domino, reduce_symmetry)
    samples_2 = sample_2_doms_last_free(
            n, rule, filter_overlap, tilt_first_domino=False,
            reduce_symmetry=False)

    samples = np.concatenate((samples_1, samples_2), axis=1)
    return samples
    # This is a previous version that I keep for the last bit (how to get
    # all combinations with numpy operations)
    #  n_pair = math.ceil(math.sqrt(n))
    #  samples_pair = sample_2_doms_last_free(
    #          n_pair, rule, filter_overlap, tilt_first_domino)
    #  samples = np.concatenate(
    #          (np.repeat(samples_pair, n_pair, axis=0),
    #           np.tile(samples_pair, (n_pair, 1))),
    #          axis=1)[:n]


sample_p_doms_straight_2_last_free = sample_3_doms_2_last_free


def sample2coords(samples, scenario, **method_params):
    """Convert the vectors generated by sample() to global domino coords.

    Parameters
    ----------
    samples : (N_samples,N_dims) ndarray
      Array of samples, as generated by sample().
    scenario : Scenario
      Domino run scenario used to generate the samples.
    method_params : dict
      Some scenarios are parametric. This is a keyword dict of such parameters.

    Returns
    -------
    coords : (N_samples,N_dominoes,3) ndarray
      Sequence of global coordinates of the dominoes for each sample.
    """
    if scenario is Scenario.TWO_DOMS_LAST_RADIAL:
        coords = np.zeros((samples.shape[0], 2,  3))
        angles = np.radians(samples[:, 1])
        coords[:, 1, 0] = samples[:, 0] * np.cos(angles)
        coords[:, 1, 1] = samples[:, 0] * np.sin(angles)
        coords[:, 1, 2] = samples[:, 1]
    elif scenario is Scenario.TWO_DOMS_LAST_FREE:
        coords = np.zeros((samples.shape[0], 2,  3))
        coords[:, 1] = samples
    elif scenario is Scenario.P_DOMS_STRAIGHT_LAST_FREE:
        nprev = method_params.pop('nprev')
        lengths = samples[:, 3] * nprev
        coords = np.zeros((samples.shape[0], nprev+2,  3))
        coords[:, :-2, 0] = np.outer(lengths, np.linspace(-1, 0, nprev+1)[:-1])
        coords[:, -1] = samples[:, :3]
    elif scenario is Scenario.THREE_DOMS_TWO_LAST_FREE:
        coords = np.zeros((samples.shape[0], 3,  3))
    elif scenario is Scenario.P_DOMS_STRAIGHT_TWO_LAST_FREE:
        nprev = method_params.pop('nprev')
        lengths = nprev * h / 3
        coords = np.zeros((samples.shape[0], nprev+3,  3))
        coords[:, :-3, 0] = np.outer(lengths, np.linspace(-1, 0, nprev+1)[:-1])
    if scenario in (
            Scenario.P_DOMS_STRAIGHT_TWO_LAST_FREE,
            Scenario.THREE_DOMS_TWO_LAST_FREE):
        coords[:, -2] = samples[:, :3]
        # Put third domino in the referential of the first domino
        angles = np.radians(samples[:, 2])  # N
        cos = np.cos(angles)  # N
        sin = np.sin(angles)  # N
        rot = np.array([[cos, -sin], [sin, cos]])  # 2x2xN
        coords[:, -1, :2] = samples[:, :2] + np.einsum(
                'ijk,kj->ki', rot, samples[:, 3:5])  # Nx2
        coords[:, -1, 2] = samples[:, 2] + samples[:, 5]
    return coords
