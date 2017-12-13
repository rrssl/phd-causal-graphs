"""
View a candidate spline from a pickled list.

Parameters
----------
pickle_path : string
  Path to the pickled list.
sid : int
  Index of the candidate spline.

"""
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    pickle_path = sys.argv[1]
    sid = int(sys.argv[2])

    with open(pickle_path, 'rb') as f:
        paths = pickle.load(f)
    path = paths[sid]

    u = np.linspace(0, 1, 100)
    x, y = splev(u, path)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.scatter(x, y, marker='+', color='r')
    ax.set_aspect('equal', 'datalim')
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
