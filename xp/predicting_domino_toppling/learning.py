"""
Regress the domino-validation function.

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
from sklearn import svm
from sklearn.externals import joblib

from config import X_MAX, Y_MAX, A_MAX


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
    svc = svm.SVC(
            kernel='rbf', gamma=1, C=1, cache_size=1024,
            #  class_weight='balanced',
            ).fit(samples, values)

    print("Score: ", svc.score(samples, values))
    root, _ = os.path.splitext(spath)
    joblib.dump(svc, root + "-classifier.pkl")


if __name__ == "__main__":
    main()
