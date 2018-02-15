import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas

sys.path.insert(0, os.path.abspath("../.."))
from xp.domino_predictors import DominoRobustness, DominoRobustness2  # noqa


FILE = "data/results.csv"
STRIDES = 6
SHAPES = [
        {'name': 'N', 'n_doms': 10},
        {'name': '[', 'n_doms': 20},
        #  {'name': 'U', 'n_doms': 11},
        ]


def load_data():
    table = pandas.read_csv(FILE)
    table = table.replace({'yes': True, 'no': False})
    return table


def grouped_bar(ax, values, barwidth, errors=None, capsize=None, labels=None,
                group_labels=None):
    group_size = len(values)
    n_groups = len(values[0])
    inter_group_width = barwidth * (group_size + 1)
    if errors is None:
        errors = [None] * group_size
    if labels is None:
        labels = [None] * group_size
    for i in range(group_size):
        x = np.arange(n_groups)*inter_group_width + i*barwidth
        ax.bar(x, values[i], barwidth, yerr=errors[i], capsize=capsize,
               label=labels[i])
    ax.set_xticks(x - (barwidth*i/2))
    ax.set_xticklabels(group_labels)


def main():
    n_shapes = len(SHAPES)
    table = load_data()
    n_samples = table.shape[0]
    means = table.mean(axis=0)
    sems = table.sem(axis=0)

    manu_means = np.array([
            means.iloc[np.arange(n_shapes) * STRIDES],
            means.iloc[np.arange(n_shapes) * STRIDES + 1]
            ])
    manu_sems = np.array([
            sems.iloc[np.arange(n_shapes) * STRIDES],
            sems.iloc[np.arange(n_shapes) * STRIDES + 1]
            ])
    barwidth = .3
    capsize = 3
    labels = ["Trial {}".format(i) for i in range(1, len(manu_means) + 1)]
    group_labels = [shape['name'] for shape in SHAPES]
    fig, ax = plt.subplots()
    grouped_bar(ax, manu_means, barwidth, manu_sems, capsize, labels,
                group_labels)
    ax.legend()
    ax.set_title(
            "Success rate: manual placement ({} samples)".format(n_samples))

    success_means = np.array([
            (manu_means[0] + manu_means[1]) / 2,
            means.iloc[np.arange(n_shapes) * STRIDES + 2],
            means.iloc[np.arange(n_shapes) * STRIDES + 4]
            ])
    success_sems = np.array([
            (manu_sems[0] + manu_sems[1]) / 2,
            sems.iloc[np.arange(n_shapes) * STRIDES + 2],
            sems.iloc[np.arange(n_shapes) * STRIDES + 4]
            ])
    labels = ["Manual", "Baseline", "Optimized"]
    fig, ax = plt.subplots()
    grouped_bar(ax, success_means, barwidth, success_sems, capsize, labels,
                group_labels)
    ax.legend()
    ax.set_title("Comparison of success rates ({} samples)".format(n_samples))

    rob_predictor = DominoRobustness2()
    rob_scores = [
            [min(rob_predictor(
                np.load("data/" + shape + "-"+ type_ + "_layout.npy")))
             for shape in group_labels]
            for type_ in ("base", "best")
            ]
    print(rob_scores)
    labels = ["Baseline", "Optimized"]
    fig, ax = plt.subplots()
    grouped_bar(ax, rob_scores, barwidth, capsize=capsize, labels=labels,
                group_labels=group_labels)
    ax.legend()
    ax.set_title("Comparison of robustness scores")

    plt.show()


if __name__ == "__main__":
    main()

