"""
Robustness and time predictors for dominoes.

"""
import os
import numpy as np
from sklearn.externals import joblib

from config import X_MAX, Y_MAX, A_MAX, MIN_SPACING, MAX_SPACING


ROB_ESTIMATOR_PATH = os.path.join(
        os.path.dirname(__file__),
        "learning_robustness/data/20171214/S2H-1000samples-classifier.pkl")
TIME_ESTIMATOR_PATH = os.path.join(
        os.path.dirname(__file__),
        "predicting_domino_timing/data/latest/samples-4D-1k-times-10-estimator.pkl")


def get_predictor_params(coords):
    coords = np.asarray(coords)
    params = np.zeros((coords.shape[0]-1, 4))
    # Change origin
    params[:, 0] = np.diff(coords[:, 0])
    params[:, 1] = np.diff(coords[:, 1])
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
    params[:, :2] = np.einsum('ijk,kj->ki', rot, params[:, :2])
    # Get relative angles in [-180, 180)
    params[:, 2] = (np.diff(coords[:, 2]) + 180) % 360 - 180
    # Symmetrize
    params[:, 2] = np.copysign(params[:, 2], params[:, 1])
    params[:, 1] = np.abs(params[:, 1])
    # Compute 4th parameter
    params[0, 3] = (MIN_SPACING + MAX_SPACING)/2  # arbitrary value
    params[1:, 3] = np.sqrt(params[:-1, 0]**2 + params[:-1, 1]**2)
    # Normalize
    params /= (X_MAX, Y_MAX, A_MAX, MAX_SPACING)
    return params


class DominoRobustness:
    def __init__(self):
        self.predictor = joblib.load(ROB_ESTIMATOR_PATH)

    def __call__(self, coords):
        return self.predictor.decision_function(coords[:, :3])


class DominoTime:
    def __init__(self):
        self.predictor = joblib.load(TIME_ESTIMATOR_PATH)

    def __call__(self, coords):
        return self.predictor.predict(coords)
