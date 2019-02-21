"""
Visualize a sketch.

Note: you can specify as many file names as you want. Parameter expansion
also works.

"""
import sys

import matplotlib.pyplot as plt
import numpy as np


MAX_NCOLS = 3


def main():
    if len(sys.argv) <= 1:
        print("Please provide a file name.")
        return
    fnames = sys.argv[1:]
    nfig = len(fnames)
    ncols = min(nfig, MAX_NCOLS)
    nrows = nfig // ncols + int(nfig % ncols > 0)
    fig = plt.figure()
    for i, fname in enumerate(fnames):
        data = np.load(fname)  # data is a list of list of pairs of floats.

        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_axis_off()
        ax.set_aspect('equal', 'datalim')
        ax.margins(.1)
        for array in data:
            ax.plot(*array.T)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
