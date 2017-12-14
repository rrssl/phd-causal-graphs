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
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

from config import X_MAX, A_MAX
sys.path.insert(0, os.path.abspath("../.."))
import xp.simulate as simu


SHOW_DECISION_FUNCTION = 1


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
        if event.mouseevent.inaxes == ax:
            idx = event.ind[0]
            print(samples[idx])
            r, a = samples[idx]
            r *= X_MAX
            a *= A_MAX
            a_ = a * math.pi / 180
            x = r * math.cos(a_)
            y = r * math.sin(a_)
            global_coords = [[0, 0, 0], [x, y, a]]
            doms_np, world = simu.setup_dominoes(
                    global_coords, _make_geom=True)
            simu.run_simu(doms_np, world, _visual=True)
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
