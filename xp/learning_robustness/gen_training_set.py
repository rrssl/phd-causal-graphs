"""
Sample the parameter space for robustness learning.

Parameters
----------
sid : int
  Sampling scenario.
nsam : int
  Number of samples.

Scenarios:
0   TwoDominoesLastRadial
1   TwoDominoesLastFree
2   DominoesStraightLastFree
3   DominoesStraightTwoLastFree
4   BallPlankDominoes

"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from xp.scenarios import SCENARIOS  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    sid = int(sys.argv[1])
    n = int(sys.argv[2])

    rule = 'H'
    samples = SCENARIOS[sid].sample_valid(n, rule=rule)
    name = "S{}{}-{}samples.npy".format(sid, rule, n)
    np.save(name, samples)


if __name__ == "__main__":
    main()
