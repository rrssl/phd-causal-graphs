"""
Evaluate different ways of estimating the toppling time of domino chains.

Parameters
----------
spath : string
  Path to the .pkl file of splines.
dpath : string
  Path to the .npz distributions along these splines.

"""
import os
import pickle
import sys
import timeit

import numpy as np
from sklearn import metrics
from tabulate import tabulate

sys.path.insert(0, os.path.abspath('..'))
from predicting_domino_timing.methods import get_methods


BASEDIR = "data/latest/"
METHODS = (
        "Simulator based",
        "Estimator (nprev=0)",
        "Estimator (nprev=1)",
        "Estimator (nprev=6)",
        "Combined estimators (0-6)",
        )
HEADERS = (
        "Method",
        "Mean absolute error",
        "R^2 score",
        "Average time per domino",
        )


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    spath = sys.argv[1]
    dpath = sys.argv[2]

    with open(spath, 'rb') as f:
        splines = pickle.load(f)
    doms = np.load(dpath)
    ns = len(splines)
    ref_method, *methods = get_methods()

    # Gather data
    ref_top_times = np.empty(ns)  # /!\ Simulator time
    ref_eval_times = np.empty(ns)  # /!\ Computation time
    est_top_times = np.empty((len(methods), ns))  # /!\ Simulator time
    est_eval_times = np.empty((len(methods), ns))  # /!\ Computation time
    ndoms = np.empty(ns)

    for i in range(len(doms.files)):
        spline = splines[i]
        distrib = doms['arr_{}'.format(i)]
        ndoms[i] = len(distrib)

        t = timeit.default_timer()
        ref_top_times[i] = ref_method(distrib, spline)
        ref_eval_times[i] = timeit.default_timer() - t

        for j, method in enumerate(methods):
            t = timeit.default_timer()
            est_top_times[j, i] = method(distrib, spline)
            est_eval_times[j, i] = timeit.default_timer() - t

    # Compute global statistics
    mean_abs_errors = np.array(
            [metrics.mean_absolute_error(ref_top_times, ett)
             for ett in est_top_times])
    r2_scores = np.array(
            [metrics.r2_score(ref_top_times, ett)
             for ett in est_top_times])
    est_eval_times_per_dom = np.array(
            [(eet / ndoms).mean()
             for eet in est_eval_times])
    all_eval_times_per_dom = np.concatenate(
            ((ref_eval_times / ndoms).mean(), est_eval_times_per_dom))

    # Show tables
    table = np.column_stack((METHODS[1:], mean_abs_errors, r2_scores))
    print(tabulate(table, headers=HEADERS[:-1]))
    table2 = np.column_stack((METHODS, all_eval_times_per_dom))
    print(tabulate(table2, headers=HEADERS[-1]))

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


if __name__ == "__main__":
    main()
