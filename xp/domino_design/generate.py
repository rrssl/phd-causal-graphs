"""
Generate a list of candidate splines from a list of input sketches.

Parameters
----------
nsplines : int
  Number of splines to generate.
skpaths : string or list of strings
  Paths to the input sketch(es).

"""
import math
import os
import pickle
import random
import sys

import numpy as np
from sklearn.externals.joblib import Parallel, delayed

import config as cfg
sys.path.insert(0, os.path.abspath("../.."))
import core.spline2d as spl  # noqa: E402
from xp.dominoes.path import DominoPathChecker  # noqa: E402


def generate_candidate_spline(sketches, size_rng, smoothing_rng):
    sketch = random.choice(sketches)
    while True:
        size_ratio = random.randint(*size_rng)
        smoothing_factor = random.uniform(*smoothing_rng)

        path = sketch[0]  # this will change when we accept several strokes
        # Translate, resize and smooth the path
        path -= path.min(axis=0)
        path *= size_ratio * math.sqrt(
                cfg.t * cfg.w / (path[:, 0].max() * path[:, 1].max()))
        spline = spl.get_smooth_path(path, s=smoothing_factor)
        tester = DominoPathChecker(spline, cfg.DOMINO_EXTENTS)
        if tester.check():
            return spline


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    nsplines = int(sys.argv[1])
    skpaths = sys.argv[2:]
    sketches = [np.load(skpath) for skpath in skpaths]

    splines = Parallel(n_jobs=cfg.NCORES)(
            delayed(generate_candidate_spline)(
                sketches,
                (cfg.MIN_SIZE_RATIO, cfg.MAX_SIZE_RATIO),
                (cfg.MIN_SMOOTHING_FACTOR, cfg.MAX_SMOOTHING_FACTOR),
                )
            for _ in range(nsplines))

    with open("candidates.pkl", 'wb') as f:
        pickle.dump(splines, f)


if __name__ == "__main__":
    main()
