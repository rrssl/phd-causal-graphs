"""
Build the various statistics used to compare the domino distribution methods.

"""
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


MAKE_FIGURES = True
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
            valarray[:, 0].mean(),
            valarray[:, 1].mean(),
            valarray[:, 2].mean(),
            overallarray.mean(),
            #  sum(ndomarray == 1) / len(ndomarray),
            (timearray / ndomarray).mean() * 1000,
            ])
    table2 = []
    for method, valfile in zip(METHODS, VAL_FILES):
        domarrays = np.load(domfile)
        valarray = np.load(valfile)
        table2.append([
            method,
            valarray[:, 3].mean(),
            valarray[:, 4].mean(),
            valarray[:, 5].mean(),
            valarray[:, 6].mean(),
            ])
    print(tabulate(table, headers=HEADERS))
    print('\n')
    print(tabulate(table2, headers=HEADERS2))

    if not MAKE_FIGURES:
        return
    # First figure
    fig, ax = plt.subplots(figsize=(9, 6))
    index = np.arange(4)
    bar_width = .1
    maxtime = max(time for *_, time in table)
    for i, (method, *data) in enumerate(table):
        ax.bar(index + i * bar_width, data[:3] + [data[4] / maxtime],
               bar_width, label=method)
    ax.set_xticks(index + bar_width*(len(table)-1)/2)
    ax.set_xticklabels(HEADERS[1:4] + (HEADERS[5] + "\n(rel. to max)",))
    ax.set_ylabel("Percentage of curves following the criterion")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.05),
              borderaxespad=2, ncol=2)
    ax.set_title("Comparison of binary validity criteria and time")
    plt.savefig("validity.png", bbox_inches='tight')
    # Second figure
    fig, ax = plt.subplots()
    x = (0, .05, .1, .15)
    for method, *data in table2:
        ax.plot(x, data, label=method, marker='o')
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Percentage of error in domino placement")
    ax.set_ylabel("Fraction of the path that toppled")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_title("Comparison of success fraction as a function of error")
    fig.tight_layout()
    plt.savefig("uncertainty.png")

if __name__ == "__main__":
    main()
