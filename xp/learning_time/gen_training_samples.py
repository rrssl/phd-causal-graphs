"""
Generate the custom distributions used to train the predictor.

Each domino pair is preceded by N equally spaced dominoes. The value of this
spacing is added to the training.

Parameters
----------
n : int
  Number of samples.

"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from xp.sampling_methods import sample4D


def main():
    if len(sys.argv) <= 1:
        print(__doc__)
        return
    n = int(sys.argv[1])

    samples = sample4D(n, filter_overlap=True, tilt_domino=False)

    name = "samples-4D.npy"
    np.save(name, samples)


if __name__ == "__main__":
    main()
