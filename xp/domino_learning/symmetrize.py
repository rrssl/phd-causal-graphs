"""
Symmetrize the samples and simulation results.

Parameters
----------
spath : string
  Path to the samples.
vpath : string
  Path to the results.

"""
import os
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    vpath = sys.argv[2]

    samples = np.load(spath)
    if samples.shape[1] > 3:
        print("No method implemented for this number of dimensions.")
        return
    values = np.load(vpath)

    samples_sym = samples.tolist()
    values_sym = values.tolist()
    if samples.shape[1] == 2:
        for s, v in zip(samples, values):
            if s[1]:
                samples_sym.append([s[0], -s[1]])
                values_sym.append(v)
    elif samples.shape[1] == 3:
        for s, v in zip(samples, values):
            if s[1]:
                samples_sym.append([s[0], -s[1], -s[2]])
                values_sym.append(v)

    root, _ = os.path.splitext(spath)
    np.save(root + "-sym.npy", samples_sym)
    root, _ = os.path.splitext(vpath)
    np.save(root + "-sym.npy", values_sym)


if __name__ == "__main__":
    main()
