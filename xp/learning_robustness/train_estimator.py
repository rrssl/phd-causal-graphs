"""
Train the domino-topples classifier.

Parameters
----------
spath : string
  Path to the samples.
vpath : string
  Path to the results.

"""
import os
import sys

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from config import X_MAX, Y_MAX, A_MAX, NCORES


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    vpath = sys.argv[2]
    samples = np.load(spath)
    # Normalize
    if samples.shape[1] == 2:
        den = (X_MAX, A_MAX)
    elif samples.shape[1] == 3:
        den = (X_MAX, Y_MAX, A_MAX)
    else:
        print("Cannot normalize for this number of dimensions.")
        return
    samples /= den
    values = np.load(vpath)

    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-3, 3, 7)
    class_weight_options = [None, 'balanced']
    param_grid = {
            'gamma': gamma_range,
            'C': C_range,
            'class_weight': class_weight_options
            }
    grid = GridSearchCV(SVC(kernel='rbf', random_state=123),
                        param_grid=param_grid,
                        n_jobs=NCORES)
    grid.fit(samples, values)
    print("The best parameters are {} with a score of {}".format(
        grid.best_params_, grid.best_score_))
    #  svc = svm.SVC(
    #          kernel='rbf', gamma=1, C=1, cache_size=1024,
    #          #  class_weight='balanced',
    #          ).fit(samples, values)
    #  print("Score: ", svc.score(samples, values))
    root, _ = os.path.splitext(spath)
    joblib.dump(grid.best_estimator_, root + "-classifier.pkl")


if __name__ == "__main__":
    main()
