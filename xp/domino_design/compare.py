"""
Build the various statistics used to compare the domino distribution methods.

"""
import numpy as np
from tabulate import tabulate


METHODS = (
        "Equal spacing",
        "Minimal spacing",
        "Simul. based inc. random search",
        "Classifier based incremental",
        "Classifier based batch (n=2)",
        "Classifier based batch (n=3)",
        )
HEADERS = (
        "Method",
        "Fully covered path",
        "No overlap",
        "All dominoes topple",
        "Overall success",
        #  "False start",
        "Success fraction",
        "Time/domino (ms)",
        )
FILES_PREFIX = "data/20170909-1/"
DOM_FILES = (
        FILES_PREFIX + "candidates-dominoes-method_{}.npz".format(i)
        for i in range(1, len(METHODS)+1)
        )
VAL_FILES = (
        FILES_PREFIX + "candidates-dominoes-method_{}-validity.npy".format(i)
        for i in range(1, len(METHODS)+1)
        )
TIME_FILES = (
        FILES_PREFIX + "candidates-times-method_{}.npy".format(i)
        for i in range(1, len(METHODS)+1)
        )


def main():
    table = []
    for method, domfile, valfile, timefile in zip(
            METHODS, DOM_FILES, VAL_FILES, TIME_FILES):
        domarrays = np.load(domfile)
        ndomarray = np.array([len(domarrays['arr_{}'.format(i)])
                     for i in range(len(domarrays.files))])
        valarray = np.load(valfile)
        overallarray = np.logical_and(
                valarray[:, 0],
                np.logical_and(valarray[:, 1], valarray[:, 2]))
        timearray = np.load(timefile)
        table.append([
            method,
            sum(valarray[:, 0]) / len(valarray),
            sum(valarray[:, 1]) / len(valarray),
            sum(valarray[:, 2]) / len(valarray),
            sum(overallarray) / len(overallarray),
            #  sum(ndomarray == 1) / len(ndomarray),
            sum(valarray[:, 3]) / len(valarray),
            (timearray / ndomarray).mean() * 1000,
            ])

    print(tabulate(table, headers=HEADERS))


if __name__ == "__main__":
    main()
