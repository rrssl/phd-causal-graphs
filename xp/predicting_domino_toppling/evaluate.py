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
from sklearn import metrics

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl

sys.path.insert(0, os.path.abspath(".."))
from predicting_domino_toppling.config import X_MAX, Y_MAX, A_MAX
from predicting_domino_toppling.functions import get_rel_coords


def eval_pairs_in_distrib(u, spline, classifier):
    # Normalize
    xya = get_rel_coords(u, spline) / (X_MAX, Y_MAX, A_MAX)
    # Evaluate
    return classifier.predict(xya)


def get_estimated_toppling_fraction(u, spline, pairwise_topple):
    n = len(pairwise_topple)
    try:
        # Find the first domino that didn't topple.
        idx = next(i for i in range(n) if not pairwise_topple[i])
    except StopIteration:
        # All dominoes toppled
        idx = n

    return np.asscalar(spl.arclength(spline, u[idx]) / spl.arclength(spline))


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
    simulated_failpoint = []
    predicted_failpoint = []
    all_topple = []
    all_topple_estimated = []
    for dominoes, validities in zip(methods_dominoes, methods_validities):
        for i, spline in enumerate(splines):
            no_overlap = validities[i, 1]
            if no_overlap:
                u = dominoes['arr_{}'.format(i)]
                lengths.append(len(u))
                # Get toppling results obtained from simulation
                all_topple.append(validities[i, 2])
                simulated_failpoint.append(validities[i, 3])
                # Estimate toppling with classifier
                pairwise_topple = eval_pairs_in_distrib(u, spline, classifier)
                predicted_failpoint.append(get_estimated_toppling_fraction(
                        u, spline, pairwise_topple))
                all_topple_estimated.append(pairwise_topple.all())
    abs_errors = np.abs(np.subtract(simulated_failpoint, predicted_failpoint))

    # Confusion matrix
    conf_mat = metrics.confusion_matrix(all_topple, all_topple_estimated)
    print("Classifier performance")
    print("Confusion matrix:\n", conf_mat)
    (tn, fp), (fn, tp) = conf_mat
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / len(all_topple)
    print("Precision = {}, recall = {}, accuracy = {}".format(
        precision, recall, accuracy))

    print("\nClassifier performance over entire chains")
    r2_score = metrics.r2_score(simulated_failpoint, predicted_failpoint)
    mean_abserror = metrics.mean_absolute_error(
            simulated_failpoint, predicted_failpoint)
    med_abserror = metrics.median_absolute_error(
            simulated_failpoint, predicted_failpoint)
    print("R^2 score: ", r2_score)
    print("Mean absolute error: ", mean_abserror)
    print("Median absolute error: ", med_abserror)

    # Estimate correlation between chain length and estimation error
    print("\nCorrelation between chain length and estimation error")
    slope, intercept, r_value, p_value, std_err = linregress(
            lengths, abs_errors)
    x = np.array([0, max(lengths)])
    y = x * slope + intercept
    print("Correlation coefficient: ", r_value)
    print("p-value: ", p_value)
    print("Standard error of the estimate: ", std_err)

    # Bin results for bar plot
    binsize = 5
    bins = list(range(0, max(lengths), binsize))
    inds = np.digitize(lengths, bins) - 1
    abs_errors = np.array(abs_errors)
    mean_abs_errors = [abs_errors[inds == i].mean() if (inds == i).any() else 0
                       for i in range(len(bins))]

    fig, ax = plt.subplots()
    ax.bar(np.array(bins)+binsize/2, mean_abs_errors, width=binsize)
    ax.plot(x, y, c='r')
    ax.set_ylim(-0.01, 1)
    ax.set_xlabel("Length of the chain")
    ax.set_ylabel("Mean absolute error in position of the failure point")
    plt.show()


if __name__ == "__main__":
    main()
