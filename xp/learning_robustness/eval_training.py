"""
Evaluate the estimator on a new dataset.

Parameters
----------
cpath : string
  Path to the .pkl classifier.
spath : string
  Path to the .npy file of samples (in feature space).
lpath : string
  Path to the .npy file of labels.

"""
import sys

import numpy as np
from sklearn.externals import joblib


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    cpath, spath, lpath = sys.argv[1:]

    clf = joblib.load(cpath)
    samples = np.load(spath)
    labels = np.load(lpath)
    score = clf.score(samples, labels)
    print("Score on the set: {}".format(score))


if __name__ == "__main__":
    main()
