"""
Generate simple splines

"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


def viewspline(spline):
    fig, ax = plt.subplots()
    u = np.linspace(0, 1)
    x, y = spl.splev(u, spline)
    ax.plot(x, y)
    plt.show()


def main():
    straight_path = np.tile(np.linspace(0, 1, 4), (2, 1)).T
    straight_spline = spl.get_smooth_path(straight_path, s=0)

    splines = [straight_spline]

    with open("simplepaths.pkl", 'wb') as f:
        pickle.dump(splines, f)


if __name__ == "__main__":
    main()
