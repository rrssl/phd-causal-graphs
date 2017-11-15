"""
Calibrate the simulator.

"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas

sys.path.insert(0, os.path.abspath("../.."))
from xp.simulate import setup_dominoes_from_path


MAKE_VISUALS = 0
BASES = ["data/20171104/I", "data/20171104/U", "data/20171104/O"]
SPLINE_EXT = ".pkl"
DOMS_EXT = "-doms.npz"
REF_TIMES_EXT = "-groundtruth.txt"


def run_simu(doms_np, world, timestep, physical_params, _visual=False):
    pass


def main():
    # Load all files
    data_dicts = []
    for base in BASES:
        # Spline
        with open(base + SPLINE_EXT, 'rb') as f:
            spline = pickle.load(f)[0]
        # Distribution
        doms = np.load(base + DOMS_EXT)['arr_0']
        # Times
        table = pandas.read_csv(base + REF_TIMES_EXT)
        pen_times = (table['pen_leaves'] - table['pen_touches']) / table['fps']
        run_times = (
                table['dn-1_touches_dn'] - table['d1_touches_d2']
                ) / table['fps']
        last_topple_times = (
                table['dn_touches_floor'] - table['dn-1_touches_dn']
                ) / table['fps']
        ref_times = [pen_times.mean(),
                     run_times.mean(),
                     last_topple_times.mean()]
        ref_times_std = [pen_times.std(),
                         run_times.std(),
                         last_topple_times.std()]

        data_dicts.append({
            'spline': spline,
            'doms': doms,
            'ref_times': ref_times,
            'ref_times_std': ref_times_std,
            'n_trials': table.shape[0]
            })
    n_shapes = len(data_dicts)
    # Show statistics of ground truth times
    if MAKE_VISUALS:
        fig, ax = plt.subplots(figsize=plt.figaspect(.5))
        barwidth = .5
        inter_plot_width = barwidth * (n_shapes + 1)
        shapes = ("$I$", "$U$", "$\Omega$")
        for i, dd in enumerate(data_dicts):
            x = np.arange(n_shapes)*inter_plot_width + i*barwidth
            y = dd['ref_times']
            yerr = dd['ref_times_std']
            label = "{} shape ($N_{{dom}}={}$, $N_{{trials}}={}$)".format(
                    shapes[i], len(dd['doms']), dd['n_trials'])
            ax.bar(x, y, barwidth, yerr=yerr, capsize=3, label=label)
        ax.set_title("Duration of some events for each path shape (mean+std)")
        ax.set_xticks(x-(barwidth*i/2))
        ax.set_xticklabels(("Initial push",
                            "From first to last collision",
                            "Last topple"))
        ax.set_ylabel("Time (s)")
        ax.legend()
        plt.show()
    # Initialize optimization
    n_events = 2
    A_ref = np.array([dd['ref_times'][1:] for dd in data_dicts])
    E = np.zeros((n_shapes, n_events))

    def objective(x):
        # Re-setup the world, or just change the parameters?
        # Compute the N_shapes-by-N_events distance matrix
        E[0] = A[0] - run_simu()
        E[1] = A[1] - run_simu()
        E[2] = A[2] - run_simu()

        return np.sum(E**2)

    # Visualize energy

    # Run optimization

    # Visualize simulations

    # Save parameters


if __name__ == "__main__":
    main()
