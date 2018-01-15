"""
Robustness and time predictors for dominoes.

"""
import os

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.externals import joblib

from .config import X_MAX, Y_MAX, A_MAX, MIN_SPACING, MAX_SPACING


ROB_ESTIMATOR_PATH = os.path.join(
        os.path.dirname(__file__),
        "learning_robustness/data/20171220/S2H-1000samples-classifier.pkl")
ROB_ESTIMATOR2_PATH = os.path.join(
        os.path.dirname(__file__),
        "learning_robustness/data/20171220/S4H-1000samples-classifier.pkl")
TIME_ESTIMATOR_PATH = os.path.join(
        os.path.dirname(__file__),
        "predicting_domino_timing/data/latest/"
        "samples-4D-1k-times-10-estimator.pkl")


def get_local_coords(coords):
    coords = np.asarray(coords)
    local_coords = np.zeros((coords.shape[0]-1, 3))
    # Change origin
    local_coords[:, :2] = np.diff(coords[:, :2], axis=0)
    # Rotate by -a_i-1
    # In case you scratch your head on this one, this is equivalent to
    # a = -head_rad[:-1]; c = cos(a); s = sin(a); rot = [[c, -s], [s, c]]
    # but more efficient because we negate once (in rot) instead of twice.
    head_rad = np.radians(coords[:, 2])
    cos = np.cos(head_rad[:-1])
    sin = np.sin(head_rad[:-1])
    rot = np.array([[cos, sin], [-sin, cos]])
    # Another tough one. This Z = einsum('ijk,kj->ki', X, Y) is equivalent to
    # for i in range(X.shape[0]):
    #     for k in range(X.shape[2]):
    #         total = 0
    #         for j in range(X.shape[1]):
    #             total += X[i,j,k] * Y[k,j]
    #         Z[k,i] = total
    # Note how the indices follow the einsum string.
    local_coords[:, :2] = np.einsum('ijk,kj->ki', rot, local_coords[:, :2])
    # Get relative angles in [-180, 180)
    local_coords[:, 2] = (np.diff(coords[:, 2]) + 180) % 360 - 180
    return local_coords


class DominoRobustness:
    def __init__(self, estimator='default'):
        if isinstance(estimator, str):
            if estimator == 'default':
                self.estimator = joblib.load(ROB_ESTIMATOR_PATH)
            else:
                self.estimator = joblib.load(estimator)
        elif isinstance(estimator, BaseEstimator):
            self.estimator = estimator

    @staticmethod
    def _transform(coords):
        """Convert global coordinates to valid parameters for the estimator."""
        params = get_local_coords(coords)
        # Symmetrize
        params[:, 2] = np.copysign(params[:, 2], params[:, 1])
        params[:, 1] = np.abs(params[:, 1])
        return params

    def __call__(self, coords):
        return self.estimator.decision_function(self._transform(coords))


class DominoRobustness2:
    def __init__(self, estimator='default'):
        if isinstance(estimator, str):
            if estimator == 'default':
                self.estimator = joblib.load(ROB_ESTIMATOR2_PATH)
            else:
                self.estimator = joblib.load(estimator)
        elif isinstance(estimator, BaseEstimator):
            self.estimator = estimator

    @staticmethod
    def _transform(coords):
        """Convert global coordinates to valid parameters for the estimator."""
        params = get_local_coords(coords)
        params = np.concatenate((params[:-1], params[1:]), axis=1)
        # Symmetrize
        neg = np.where(params[:, 1] < 0)[0][:, np.newaxis]  # N_negx1
        params[neg, [1, 2, 4, 5]] *= -1
        return params

    def __call__(self, coords):
        return self.estimator.decision_function(self._transform(coords))


class DominoTime:
    def __init__(self, estimator='default'):
        if isinstance(estimator, str):
            if estimator == 'default':
                self.estimator = joblib.load(TIME_ESTIMATOR_PATH)
            else:
                self.estimator = joblib.load(estimator)
        elif isinstance(estimator, BaseEstimator):
            self.estimator = estimator

    @staticmethod
    def _transform(coords):
        """Convert global coordinates to valid parameters for the estimator."""
        params = np.empty((coords.shape[0]-1, 4))
        params[:, :3] = get_local_coords(coords)
        # Symmetrize
        params[:, 2] = np.copysign(params[:, 2], params[:, 1])
        params[:, 1] = np.abs(params[:, 1])
        # Compute 4th parameter
        params[0, 3] = (MIN_SPACING + MAX_SPACING)/2  # arbitrary value
        params[1:, 3] = np.sqrt(params[:-1, 0]**2 + params[:-1, 1]**2)
        # Normalize
        params /= (X_MAX, Y_MAX, A_MAX, MAX_SPACING)
        return params

    def __call__(self, coords):
        return self.estimator.predict(self._transform(coords))
