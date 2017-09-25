"""
Evaluate the ability of the classifier to assess the failure of a chain.

Compute the mean absolute difference between the failure point obtained
from simulation, and the one obtained from the classifier, as a function of
the number of elements in the chain.

Parameters
----------
cpath : string
  Path to the .pkl classifier.
spath : string
  Path to the .pkl file of splines.

"""
import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn.externals import joblib

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl

sys.path.insert(0, os.path.abspath(".."))
from domino_learning.config import X_MAX, Y_MAX, A_MAX


def eval_pairs_in_distrib(u, spline, classifier):
    # Get local Cartesian coordinates
    # Change origin
    xi, yi = spl.splev(u, spline)
    xi = xi[1:] - xi[:-1]
    yi = yi[1:] - yi[:-1]
    # Rotate by -a_i-1
    ai = spl.splang(u, spline)
    ci_ = np.cos(ai[:-1])
    si_ = np.sin(ai[:-1])
    xi = xi*ci_ + yi*si_
    yi = -xi*si_ + yi*ci_
    # Get relative angles
    ai = np.degrees(ai[1:] - ai[:-1])
    ai = (ai + 180) % 360 - 180  # Convert from [0, 360) to [-180, 180)
    # Symmetrize
    ai = np.copysign(ai, yi)
    yi = abs(yi)
    # Normalize
    xi /= X_MAX
    yi /= Y_MAX
    ai /= A_MAX
    # Evaluate
    return classifier.predict(np.column_stack((xi, yi, ai)))


def get_estimated_toppling_fraction(u, spline, classifier):
    pairs_results = eval_pairs_in_distrib(u, spline, classifier)
    n = len(pairs_results)
    try:
        # Find the first domino that didn't topple.
        idx = next(i for i in range(n) if pairs_results[i] == -1)
    except StopIteration:
        # All dominoes toppled
        idx = n

    return spl.arclength(spline, u[idx]) / spl.arclength(spline)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cpath = sys.argv[1]
    spath = sys.argv[2]
    root, _ = os.path.splitext(spath)
    dpaths = glob.glob(root + "-dominoes-method_*.npz")
    dpaths.sort()
    vpaths = glob.glob(root + "-dominoes-method_*-validity.npy")
    vpaths.sort()

    classifier = joblib.load(cpath)
    with open(spath, 'rb') as f:
        splines = pickle.load(f)
    methods_dominoes = [np.load(dpath) for dpath in dpaths]
    methods_validities = [np.load(vpath) for vpath in vpaths]

    lengths = []
    abs_errors = []
    for dominoes, validities in zip(methods_dominoes, methods_validities):
        for i, spline in enumerate(splines):
            no_overlap = validities[i, 1]
            if no_overlap:
                toppling_fraction = validities[i, 3]
                u = dominoes['arr_{}'.format(i)]
                estimated_toppling_fraction = get_estimated_toppling_fraction(
                        u, spline, classifier)
                abs_error = abs(toppling_fraction-estimated_toppling_fraction)
                lengths.append(len(u))
                abs_errors.append(np.asscalar(abs_error))

    binsize = 5
    bins = list(range(0, max(lengths), binsize))
    inds = np.digitize(lengths, bins) - 1
    abs_errors = np.array(abs_errors)
    mean_abs_errors = [abs_errors[inds == i].mean() if (inds == i).any() else 0
                       for i in range(len(bins))]

    slope, intercept, r_value, p_value, std_err = linregress(
            lengths, abs_errors)
    x = np.array([0, max(lengths)])
    y = x * slope + intercept
    print("Correlation coefficient: ", r_value)
    print("p-value: ", p_value)
    print("Standard error of the estimate: ", std_err)

    fig, ax = plt.subplots()
    ax.bar(np.array(bins)+binsize/2, mean_abs_errors, width=binsize)
    ax.plot(x, y, c='r')
    ax.set_ylim(-0.01, 1)
    ax.set_xlabel("Length of the chain")
    ax.set_ylabel("Mean absolute error in position of the failure point")
    plt.show()


if __name__ == "__main__":
    main()
