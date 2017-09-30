"""
Visualize the evaluation results for each estimation method.

"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn import metrics
from tabulate import tabulate


BASEDIR = "data/latest/"
METHODS = (
        "Simulator based (ts=1/500)",
        "Simulator based (ts=1/60)",
        "Estimator (nprev=0)",
        "Estimator (nprev=1)",
        "Estimator (nprev=3)",
        "Estimator (nprev=4)",
        "Estimator (nprev=5)",
        "Estimator (nprev=6)",
        "Estimator (nprev=7)",
        "Combined estimators (0-6)",
        )
HEADERS = (
        "Method",
        "Mean absolute error",
        "R^2 score",
        "Mean computation time",
        )


def main():
    phys_times_files = glob.glob(BASEDIR + "method_*-phys-times.npy")
    phys_times_files.sort()
    comp_times_files = glob.glob(BASEDIR + "method_*-comp-times.npy")
    comp_times_files.sort()

    phys_times = [np.load(phys_times_file)
                  for phys_times_file in phys_times_files]
    comp_times = [np.load(comp_times_file)
                  for comp_times_file in comp_times_files]

    # Compute global statistics
    mean_abs_errors = np.array(
            [metrics.mean_absolute_error(phys_times[0], pt)
                for pt in phys_times[1:]])
    r2_scores = np.array(
            [metrics.r2_score(phys_times[0], pt)
                for pt in phys_times[1:]])
    mean_comp_times = [comp_time.mean() for comp_time in comp_times]

    # Show tables
    table = np.column_stack((METHODS[1:], mean_abs_errors, r2_scores))
    print('\n')
    print(tabulate(table, headers=HEADERS[:-1]))
    table2 = np.column_stack((METHODS, mean_comp_times))
    print('\n')
    print(tabulate(table2, headers=(HEADERS[0], HEADERS[-1])))
    print('\n')

    #  # First figure
    #  fig, ax = plt.subplots(figsize=(9, 6))
    #  index = np.arange(4)
    #  bar_width = .1
    #  maxtime = max(time for *_, time in table)
    #  for i, ((method, *means), err) in enumerate(zip(table, table_err)):
    #      x = index + i * bar_width
    #      y = means[:3] + [means[4] / maxtime]
    #      err = err[:3] + [err[4] / maxtime]
    #      ax.bar(x, y, bar_width, label=method)
    #      ax.errorbar(x, y, yerr=err, fmt='none', capsize=5, ecolor='k')
    #  ax.set_xticks(index + bar_width*(len(table)-1)/2)
    #  ax.set_xticklabels(HEADERS[1:4] + (HEADERS[5] + "\n(rel. to max)",))
    #  ax.set_ylabel("Percentage of curves following the criterion")
    #  handles, labels = ax.get_legend_handles_labels()
    #  ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.05),
    #            borderaxespad=2, ncol=2)
    #  ax.set_title("Comparison of regression scores between the methods")
    #  plt.savefig(BASEDIR + "methods_comparison.png", bbox_inches='tight')

    # Show absolute error as a function of chain size
    lengths = np.load(BASEDIR + "last-top-inds.npy")
    abs_errors = abs(phys_times[1:] - phys_times[0])
    fig, ax = plt.subplots()
    for ae, method in zip(abs_errors, METHODS[1:]):
        slope, intercept, r_value, p_value, std_err = linregress(
                lengths, ae)
        x = np.array([0, max(lengths)])
        y = x * slope + intercept
        ax.plot(x, y, label=method)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.05),
              borderaxespad=2, ncol=2)
    ax.set_xlabel("Length of the chain")
    ax.set_ylabel("Mean absolute error in time")
    ax.set_ylim(-0.1, 5)

    plt.savefig(BASEDIR + "error_vs_chain_length.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
