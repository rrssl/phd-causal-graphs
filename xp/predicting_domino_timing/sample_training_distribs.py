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
from random import uniform
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(".."))
from predicting_domino_timing.config import t, w, h
from predicting_domino_timing.config import X_MIN, X_MAX
from predicting_domino_timing.config import Y_MAX
from predicting_domino_timing.config import A_MIN, A_MAX
from predicting_domino_timing.config import MIN_SPACING, MAX_SPACING
from predicting_domino_toppling.functions import make_box, has_contact


def sample(n, bounds, filter_overlap=True):
    """Sample the domino-pair parameter values with 4 DoFs (relative position
    in the XY plane + relative heading angle + spacing between previous doms).

    """
    samples = np.empty((n, len(bounds)))

    if filter_overlap:
        i = 0
        while i < n:
            x = uniform(*bounds[0])
            y = uniform(*bounds[1])
            a = uniform(*bounds[2])
            s = uniform(*bounds[3])

            d1 = make_box((t, w, h), (0, 0, h/2), (0, 0, 0))
            d2 = make_box((t, w, h), (x, y, h/2), (a, 0, 0))
            if not has_contact(d1, d2):
                samples[i] = x, y, a, s
                i += 1
    else:
        samples[:, 0] = np.random.uniform(bounds[0][0], bounds[0][1], n)
        samples[:, 1] = np.random.uniform(bounds[1][0], bounds[1][1], n)
        samples[:, 2] = np.random.uniform(bounds[2][0], bounds[2][1], n)
        samples[:, 3] = np.random.uniform(bounds[3][0], bounds[3][1], n)

    return samples


def main():
    if len(sys.argv) <= 1:
        print(__doc__)
        return
    nsam = int(sys.argv[1])

    # Y is > 0 because of symmetry.
    bounds = ((X_MIN, X_MAX),
              (0, Y_MAX),
              (A_MIN, A_MAX),
              (MIN_SPACING, MAX_SPACING))
    samples = sample(nsam, bounds, filter_overlap=True)

    name = "samples-4D.npy"
    np.save(name, samples)


if __name__ == "__main__":
    main()
