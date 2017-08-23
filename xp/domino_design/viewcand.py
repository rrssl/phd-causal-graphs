"""
View a candidate path from a pickled list.

"""
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev


def main():
    path_id = int(sys.argv[1])
    pickle_path = sys.argv[2]

    with open(pickle_path, 'rb') as f:
        paths = pickle.load(f)
    path = paths[path_id]

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
