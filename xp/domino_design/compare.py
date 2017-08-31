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
        "Classifier based batch (n=4)",
        "Classifier based batch (n=5)",
        )
HEADERS = (
        "Method",
        "Fully covered path",
        "No overlap",
        "All dominoes topple",
        "Overall success",
        "False start",
        "Time/domino (ms)",
        )
DOM_FILES = (
        #  "test-100/dominoes-method_1.npz",
        #  "test-100/test-dominoes-method_2.npz",
        #  "test-100/test-dominoes-method_3.npz",
        #  "test-100/dominoes-method_4.npz",
        #  "test-100/test-dominoes-method_5.npz",
        "data/20173108-test-100/candidates-dominoes-method_{}.npz".format(i)
        for i in range(1, len(METHODS)+1)
        )
VAL_FILES = (
        #  "test-100/validity-method_1.npy",
        #  "test-100/test-dominoes-method_2-validity.npy",
        #  "test-100/test-dominoes-method_3-validity.npy",
        #  "test-100/validity-method_4.npy",
        #  "test-100/test-dominoes-method_5-validity.npy",
        "data/20173108-test-100/candidates-dominoes-method_{}-validity.npy".format(i)
        for i in range(1, len(METHODS)+1)
        )
TIME_FILES = (
        #  "test-100/times-method_1.npy",
        #  "test-100/test-times-method_2.npy",
        #  "test-100/test-times-method_3.npy",
        #  "test-100/times-method_4.npy",
        #  "test-100/test-times-method_5.npy",
        "data/20173108-test-100/candidates-times-method_{}.npy".format(i)
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
            sum(ndomarray == 1) / len(ndomarray),
            (timearray / ndomarray).mean() * 1000,
            ])

    print(tabulate(table, headers=HEADERS))


if __name__ == "__main__":
    main()
