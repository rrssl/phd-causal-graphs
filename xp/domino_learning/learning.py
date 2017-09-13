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

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.externals import joblib


def visualize(samples, values, svc):
    # Create mesh for sampling
    n = 200
    xmin = samples[:, 0].min()
    xmax = samples[:, 0].max()
    xmargin = (xmax - xmin) * .1
    ymin = samples[:, 1].min()
    ymax = samples[:, 1].max()
    ymargin = (ymax - ymin) * .1
    xx, yy = np.meshgrid(np.linspace(xmin - xmargin, xmax + xmargin, n),
                         np.linspace(ymin - ymargin, ymax + ymargin, n))
    zz = svc.predict(np.column_stack((xx.ravel(), yy.ravel())))
    zz = zz.reshape(xx.shape)
    # Plot
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.contourf(xx, yy, zz, alpha=.8)
    ax.scatter(samples[:, 0], samples[:, 1], c=values, edgecolor='.5')
    ax.set_xlabel("Normalized distance")
    ax.set_ylabel("Normalized angle")
    ax.set_title("Learning a binary classifier: SVC with RBF kernel "
                 "$(C = 1, \gamma = 1)$")
    plt.show()

    # Get decision function values
    dist = svc.decision_function(np.column_stack((xx.ravel(), yy.ravel())))
    dist = dist.reshape(xx.shape)
    # Plot
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    reg = ax.pcolormesh(xx, yy, dist, alpha=1.)
    fig.colorbar(reg)
    ax.scatter(samples[:, 0], samples[:, 1], c=values, edgecolor='.5')
    ax.set_xlabel("Normalized distance")
    ax.set_ylabel("Normalized angle")
    ax.set_title("Decision function (i.e. distance to the boundary)\n"
                 "for an SVC with RBF kernel $(C = 1, \gamma = 1)$")
    plt.show()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    vpath = sys.argv[2]
    samples = np.load(spath)
    # TODO. This is lazy. Use the actual sampling bounds.
    samples /= samples.max(axis=0)
    values = np.load(vpath)
    svc = svm.SVC(kernel='rbf', gamma=1, C=1).fit(samples, values)

    print("Score: ", svc.score(samples, values))
    root, _ = os.path.splitext(spath)
    joblib.dump(svc, root + "-classifier.pkl")

    if samples.shape[1] == 2:
        visualize(samples, values, svc)


if __name__ == "__main__":
    main()
