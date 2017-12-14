"""
Sample the parameter space for robustness learning.

Parameters
----------
scenario : int
  Sampling scenario. See sampling_methods.py for a description.
nsam : int
  Number of samples.

"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
from xp.sampling_methods import sample, Scenario


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    scenario = int(sys.argv[1])
    n = int(sys.argv[2])

    filter_rules = dict(filter_overlap=True, tilt_first_domino=True)
    samples = sample(n, scenario=Scenario(scenario), generator_rule='R',
                     filter_rules=filter_rules)
    name = "S{}-{}samples.npy".format(scenario, n)
    np.save(name, samples)


if __name__ == "__main__":
    main()
