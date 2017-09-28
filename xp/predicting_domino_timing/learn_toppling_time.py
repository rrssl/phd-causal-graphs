"""
Regress the domino timing function.

Parameters
----------
spath : string
  Path to the .npy samples.
tpath : string
  Path to the .npy times.

"""
import os
import sys
import timeit

import numpy as np
from sklearn import svm
from sklearn import neural_network
from sklearn import kernel_ridge
from sklearn.externals import joblib
import sklearn.metrics as metrics

sys.path.insert(0, os.path.abspath('..'))
from predicting_domino_timing.config import X_MAX, Y_MAX, A_MAX, MAX_SPACING


# 0 = Support Vector Machine, 1 = Kernel Ridge, 2 = Neural Network
ESTIMATOR_TYPE = 1


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    tpath = sys.argv[2]
    samples = np.load(spath)
    times = np.load(tpath)
    # Filter values
    valid = np.isfinite(times)
    samples = samples[valid]
    times = times[valid]
    print("Times mean = {}, std = {}".format(times.mean(), times.std()))
    # Normalize
    if samples.shape[1] == 2:
        den = (X_MAX, A_MAX)
    elif samples.shape[1] == 3:
        den = (X_MAX, Y_MAX, A_MAX)
    elif samples.shape[1] == 4:
        den = (X_MAX, Y_MAX, A_MAX, MAX_SPACING)
    else:
        print("Cannot normalize for this number of dimensions.")
        return
    samples /= den
    # Choose estimator
    if ESTIMATOR_TYPE == 0:
        estimator = svm.SVR(kernel='rbf', gamma=100, C=100, cache_size=1024)
    elif ESTIMATOR_TYPE == 1:
        estimator = kernel_ridge.KernelRidge(kernel='rbf', gamma=500, alpha=.1)
    elif ESTIMATOR_TYPE == 2:
        estimator = neural_network.MLPRegressor(
                hidden_layer_sizes=(30, 30, 30),
                solver='lbfgs', activation='relu',
                alpha=1e-4, max_iter=3000, tol=1e-4,
                )
    # Train the estimator
    t = timeit.default_timer()
    estimator.fit(samples, times)
    t = timeit.default_timer() - t
    print("Training time per sample: ", t / times.size)

    t = timeit.default_timer()
    predicted = estimator.predict(samples)
    t = timeit.default_timer() - t
    print("Evaluation time per sample: ", t / times.size)
    print("Explained variance score: ",
          metrics.explained_variance_score(times, predicted))
    print("Mean absolute error: ",
          metrics.mean_absolute_error(times, predicted))
    print("R2 score: ", metrics.r2_score(times, predicted))

    # Save the estimator
    root, _ = os.path.splitext(spath)
    joblib.dump(estimator, root + "-estimator.pkl")


if __name__ == "__main__":
    main()
