# coding: utf-8
"""
This is the trace of an IPython session showing that our improved tests for
path validity are actually a good predictor of whether the methods are going
to fail or not.
In other words, it shows that in many cases where methods failed, it was not
possible to succeed anyway.

"""
import numpy as np
import pickle
from sklearn.metrics import matthews_corrcoef

from generate import DominoPathTester


with open("data/test-100/candidates.pkl", 'rb') as f:
    splines = pickle.load(f)
checks = [DominoPathTester(spline).check() for spline in splines]
validities = [np.load("data/20170901-2/candidates-dominoes-method_{}-validity.npy".format(i)) for i in range(1, 6)]
validities[0][0]
overall_validities = [((validity[:, 0] + validity[:, 1] + validity[:, 2]) / 3) == 1 for validity in validities]
overall_validities[0][0]
correlations = [matthews_corrcoef(overall_validity, checks) for overall_validity in overall_validities]
correlations
