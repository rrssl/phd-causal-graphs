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
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn import metrics
from sklearn.externals import joblib

from config import NCORES
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl  # noqa
from xp.dominoes.creation import equal_spacing  # noqa
from xp.domino_predictors import DominoRobustness, DominoRobustness2  # noqa
import xp.simulate as simu  # noqa


def true_frac(a):
    first_false = np.argmin(a)
    if a[first_false]:
        return 1
    else:
        return first_false / len(a)


def compute_simulated_toppling_fraction(u, spline):
    doms_np, world = simu.setup_dominoes_from_path(u, spline)
    times = simu.run_simu(doms_np, world)
    frac = true_frac(np.isfinite(times))
    return frac


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cpath = sys.argv[1]
    spath = sys.argv[2]

    type_ = int(os.path.basename(cpath)[1])
    if type_ == 2:
        estimator = DominoRobustness(cpath)
    elif type_ == 4:
        estimator = DominoRobustness2(cpath)
    with open(spath, 'rb') as f:
        splines = pickle.load(f)

    # Generate ground truth data.
    root = os.path.splitext(spath)[0]
    filename = root + "-simu-toppling.npz"
    try:
        data = np.load(filename)
        simu_toppling = data['simu_toppling']
        distribs = [data['arr_{}'.format(i)]
                    for i in range(len(simu_toppling))]
    except FileNotFoundError:
        distribs = [equal_spacing(spline) for spline in splines]
        simu_toppling = joblib.Parallel(n_jobs=NCORES)(
                joblib.delayed(compute_simulated_toppling_fraction)(u, spline)
                for u, spline in zip(distribs, splines))
        np.savez(filename, *distribs, simu_toppling=simu_toppling)

    # Generate predictions.
    distribs_coords = [
            np.column_stack(spl.splev(u, spline) + [spl.splang(u, spline)])
            for u, spline in zip(distribs, splines)]
    pred_toppling = np.array([
            true_frac(estimator(coords) >= 0) for coords in distribs_coords])

    # Confusion matrix
    conf_mat = metrics.confusion_matrix(
            simu_toppling == 1, pred_toppling == 1)
    print("Classification performance over entire chains")
    print("Confusion matrix:\n", conf_mat)
    (tn, fp), (fn, tp) = conf_mat
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / len(splines)
    print("Precision = {}, recall = {}, accuracy = {}".format(
        precision, recall, accuracy))

    print("\nSuccess fraction estimation performance over entire chains")
    r2_score = metrics.r2_score(simu_toppling, pred_toppling)
    mean_abserror = metrics.mean_absolute_error(simu_toppling, pred_toppling)
    med_abserror = metrics.median_absolute_error(simu_toppling, pred_toppling)
    print("R^2 score: ", r2_score)
    print("Mean absolute error: ", mean_abserror)
    print("Median absolute error: ", med_abserror)

    # Estimate correlation between chain length and estimation error
    abs_errors = np.abs(np.subtract(simu_toppling, pred_toppling))
    lengths = [len(u) for u in distribs]
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
    mean_abs_errors = [abs_errors[inds == i].mean() if i in inds else 0
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
