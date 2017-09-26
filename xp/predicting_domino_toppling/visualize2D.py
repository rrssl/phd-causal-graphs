"""
Visualize a 2D sampling of interacting domino pairs.

Parameters
----------
spath : string
  Path to the samples.
vpath : string
  Path to the results.
cpath : string
  Path to the classifier.

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

from config import X_MAX, A_MAX
from config import t, w, h, density
from functions import run_domino_toppling_xp


SHOW_DECISION_FUNCTION = 0


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
    # Plot
    fig, ax = plt.subplots()

    if SHOW_DECISION_FUNCTION:
        # Get decision function values
        dist = svc.decision_function(np.column_stack((xx.ravel(), yy.ravel())))
        dist = dist.reshape(xx.shape)
        # Plot
        reg = ax.pcolormesh(xx, yy, dist, alpha=1.)
        fig.colorbar(reg)
        ax.set_title("Decision function (i.e. distance to the boundary)\n"
                     "for an SVC with RBF kernel $(C = 1, \gamma = 1)$")
    else:
        # Get classifier values
        zz = svc.predict(np.column_stack((xx.ravel(), yy.ravel())))
        zz = zz.reshape(xx.shape)
        # Plot
        ax.contourf(xx, yy, zz, alpha=.8)
        ax.set_title("Learning a binary classifier: SVC with RBF kernel "
                     "$(C = 1, \gamma = 1)$")
    ax.scatter(samples[:, 0], samples[:, 1], c=values, edgecolor='.5',
               picker=True)
    ax.set_xlabel("Normalized distance")
    ax.set_ylabel("Normalized angle")

    # Interactive point picker
    def onpick(event):
        idx = event.ind[0]
        print(samples[idx])
        d, a = samples[idx]
        d *= X_MAX
        a *= A_MAX
        mass = t * w * h * density
        run_domino_toppling_xp((t, w, h, d, 0, a, mass), 0, 0, visual=True)
    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    spath = sys.argv[1]
    vpath = sys.argv[2]
    cpath = sys.argv[3]
    samples = np.load(spath)
    # Normalize
    den = (X_MAX, A_MAX)
    samples /= den
    values = np.load(vpath)
    svc = joblib.load(cpath)
    visualize(samples, values, svc)


if __name__ == "__main__":
    main()
