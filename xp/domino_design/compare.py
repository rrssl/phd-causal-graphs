"""
Build the various statistics used to compare the domino distribution methods.

"""
import numpy as np
import scipy.stats
from tabulate import tabulate
import matplotlib.pyplot as plt


MAKE_FIGURES = 1
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
        "Time/domino (ms)",
        )
HEADERS2 = (
        "Method",
        "Success fraction",
        "-- with 5% error",
        "-- with 10% error",
        "-- with 15% error",
        )
HEADERS3 = (
        "Method",
        "Density",
        )
FILES_PREFIX = "data/20170909-2/"
DOM_FILES = [
        FILES_PREFIX + "candidates-dominoes-method_{}.npz".format(i)
        for i in range(1, len(METHODS)+1)
        ]
VAL_FILES = [
        FILES_PREFIX + "candidates-dominoes-method_{}-validity.npy".format(i)
        for i in range(1, len(METHODS)+1)
        ]
TIME_FILES = [
        FILES_PREFIX + "candidates-times-method_{}.npy".format(i)
        for i in range(1, len(METHODS)+1)
        ]
MET_FILES = [
        FILES_PREFIX + "candidates-dominoes-method_{}-metrics.npy".format(i)
        for i in range(1, len(METHODS)+1)
        ]
ERR_FUNC = scipy.stats.sem
#  ERR_FUNC = np.std


def main():
    table = []
    table_err = []
    table2 = []
    table2_err = []
    table3 = []
    table3_err = []
    for method, domfile, valfile, timefile, metfile in zip(
            METHODS, DOM_FILES, VAL_FILES, TIME_FILES, MET_FILES):
        domarrays = np.load(domfile)
        ndomarray = np.array([len(domarrays['arr_{}'.format(i)])
                              for i in range(len(domarrays.files))])
        valarray = np.load(valfile)
        overallarray = np.logical_and(
                valarray[:, 0],
                np.logical_and(valarray[:, 1], valarray[:, 2]))
        timearray = np.load(timefile)
        metarray = np.load(metfile)
        table.append([
            method,
            valarray[:, 0].mean(),
            valarray[:, 1].mean(),
            valarray[:, 2].mean(),
            overallarray.mean(),
            #  (ndomarray == 1) .mean(),
            (timearray / ndomarray).mean() * 1000,
            ])
        table_err.append([
            ERR_FUNC(valarray[:, 0]),
            ERR_FUNC(valarray[:, 1]),
            ERR_FUNC(valarray[:, 2]),
            ERR_FUNC(overallarray),
            #  ERR_FUNC((ndomarray == 1)),
            ERR_FUNC((timearray / ndomarray) * 1000),
            ])
        table2.append([
            method,
            valarray[:, 3].mean(),
            valarray[:, 4].mean(),
            valarray[:, 5].mean(),
            valarray[:, 6].mean(),
            ])
        table2_err.append([
            ERR_FUNC(valarray[:, 3]),
            ERR_FUNC(valarray[:, 4]),
            ERR_FUNC(valarray[:, 5]),
            ERR_FUNC(valarray[:, 6]),
            ])
        table3.append([
            method,
            metarray[:, 0].mean(),
            ])
        table3_err.append([
            ERR_FUNC(metarray[:, 0]),
            ])
    print(tabulate(table, headers=HEADERS))
    print('\n')
    print(tabulate(table2, headers=HEADERS2))
    print('\n')
    print(tabulate(table3, headers=HEADERS3))

    if not MAKE_FIGURES:
        return
    # First figure
    fig, ax = plt.subplots(figsize=(9, 6))
    index = np.arange(4)
    bar_width = .1
    maxtime = max(time for *_, time in table)
    for i, ((method, *means), err) in enumerate(zip(table, table_err)):
        x = index + i * bar_width
        y = means[:3] + [means[4] / maxtime]
        err = err[:3] + [err[4] / maxtime]
        ax.bar(x, y, bar_width, label=method)
        ax.errorbar(x, y, yerr=err, fmt='none', capsize=5, ecolor='k')
    ax.set_xticks(index + bar_width*(len(table)-1)/2)
    ax.set_xticklabels(HEADERS[1:4] + (HEADERS[5] + "\n(rel. to max)",))
    ax.set_ylabel("Percentage of curves following the criterion")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.05),
              borderaxespad=2, ncol=2)
    ax.set_title("Comparison of binary validity criteria + time")
    plt.savefig(FILES_PREFIX + "validity.png", bbox_inches='tight')
    # Second figure
    fig, ax = plt.subplots()
    x = (0, .05, .1, .15)
    for (method, *means), err in zip(table2, table2_err):
        ax.errorbar(x, means, yerr=err, fmt='-o', capsize=5, label=method)
    ax.set_xticks(x)
    ax.set_xlabel("Percentage of error in domino placement")
    ax.set_ylabel("Fraction of the path that toppled")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("Comparison of success fraction as a function of error")
    plt.savefig(FILES_PREFIX + "uncertainty.png", bbox_inches='tight')
    # Third figure
    fig, ax = plt.subplots()
    index = np.arange(1)
    bar_width = .01
    for i, ((method, *means), err) in enumerate(zip(table3, table3_err)):
        ax.bar(index + i*bar_width, means[:3], bar_width, label=method)
        ax.errorbar(index + i*bar_width, means[:3], yerr=err[:3],
                    fmt='none', capsize=5, ecolor='k')
    #  ax.set_xlim(-.03, .05)
    ax.set_ylim(0, 1)
    ax.set_xticks(index + bar_width*(len(table)-1)/2)
    ax.set_xticklabels(HEADERS3[1:])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("Metrics comparisons for different spacing values")
    plt.savefig(FILES_PREFIX + "metrics.png", bbox_inches='tight')


if __name__ == "__main__":
    main()
