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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import NCORES


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    vpath = sys.argv[2]
    samples = np.load(spath)
    values = np.load(vpath)

    pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', random_state=123, cache_size=512),
            )

    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-3, 3, 7)
    class_weight_options = [None, 'balanced']
    param_grid = {
            'svc__gamma': gamma_range,
            'svc__C': C_range,
            'svc__class_weight': class_weight_options
            }

    grid = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=NCORES)
    grid.fit(samples, values)
    print("The best parameters are {} with a score of {}".format(
        grid.best_params_, grid.best_score_))

    root, _ = os.path.splitext(spath)
    joblib.dump(grid.best_estimator_, root + "-classifier.pkl")


if __name__ == "__main__":
    main()
