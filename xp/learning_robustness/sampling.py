"""
Sampling the parameter space.

Parameters
----------
nsam : int
  Number of samples.
ndof : int, optional
  Number of degrees of freedom. Defaults to 3 (relative x, y, and angle).

"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from xp.sampling_methods import sample2D, sample3D


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    nsam = int(sys.argv[1])
    ndim = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if ndim == 2:
        samples = sample2D(nsam, filter_overlap=True, tilt_domino=True)
    elif ndim == 3:
        samples = sample3D(nsam, filter_overlap=True, tilt_domino=True)
    else:
        print("No method implemented for this number of dimensions.")
        return

    name = "samples-{}D.npy".format(ndim)
    np.save(name, samples)


if __name__ == "__main__":
    main()
