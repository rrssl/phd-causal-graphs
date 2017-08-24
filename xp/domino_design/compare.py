"""
Build the various statistics used to compare the domino distribution methods.

Parameters
----------
datapath : string
  Path where the data lies.

"""
import glob
import os
import sys

import numpy as np
from tabulate import tabulate


METHODS = (
        "Equal spacing",
        "Minimal spacing",
        "Classifier based incremental",
        "Simul. based inc. random search",
        )
HEADERS = (
        "Method",
        "Success rate",
        "Time/domino (ms)",
        "False start rate"
        )


def main():
    if len(sys.argv) > 2:
        print(__doc__)
        return
    datapath = sys.argv[1]
    dominoes_files = glob.glob(os.path.join(datapath, "dominoes*.npz"))
    validity_files = glob.glob(os.path.join(datapath, "validity*.npy"))
    times_files = glob.glob(os.path.join(datapath, "times*.npy"))
    dominoes_files.sort()
    validity_files.sort()
    times_files.sort()

    table = []
    for method, domfile, valfile, timefile in zip(
            METHODS, dominoes_files,validity_files, times_files):
        domarrays = np.load(domfile)
        ndomarray = np.array([len(domarrays['arr_{}'.format(i)])
                     for i in range(len(domarrays.files))])
        valarray = np.load(valfile)
        timearray = np.load(timefile)
        table.append([
            method,
            sum(valarray) / len(valarray),
            (timearray / ndomarray).mean() * 1000,
            sum(ndomarray == 1) / len(ndomarray),
            ])

    print(tabulate(table, headers=HEADERS))


if __name__ == "__main__":
    main()
