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
from sklearn.externals import joblib
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import sklearn.metrics as metrics

from config import X_MAX, Y_MAX, A_MAX, MAX_SPACING, NCORES


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
        param_grid = {
                'gamma': np.logspace(-3, 3, 7),
                'C': np.logspace(-3, 3, 7),
                }
        estimator = SVR(kernel='rbf')
    elif ESTIMATOR_TYPE == 1:
        param_grid = {
                'gamma': np.logspace(-3, 3, 7),
                'alpha': np.logspace(-3, 3, 7),
                }
        estimator = KernelRidge(kernel='rbf')
    elif ESTIMATOR_TYPE == 2:
        param_grid = {
                'hidden_layer_sizes': [(10,)*n for n in range(1,4)],
                'alpha': np.logspace(-4, 0, 5),
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'adam'],
                }
        estimator = MLPRegressor(
                max_iter=1000, random_state=123)
    grid = GridSearchCV(estimator, param_grid=param_grid, n_jobs=NCORES)
    grid.fit(samples, times)
    print("The best parameters are {} with a score of {}".format(
        grid.best_params_, grid.best_score_))
    estimator = grid.best_estimator_
    # Train the estimator
    #  t = timeit.default_timer()
    #  estimator.fit(samples, times)
    #  t = timeit.default_timer() - t
    #  print("Training time per sample: ", t / times.size)

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
    root, _ = os.path.splitext(tpath)
    joblib.dump(estimator, root + "-estimator.pkl")


if __name__ == "__main__":
    main()
