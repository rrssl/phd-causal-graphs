"""
Generate the visuals.

Parameters
----------
spath : string
  Path to the .pkl file of candidate splines.

"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl


USE_KRV_INSTEAD_OF_ANGLE = 1
COMBINE_PLOTS = 0
SHOW_ANGLES = 0
SHOW_ARCLENGTHS = 1


def get_local_variables(spline, dominoes, times):
    maxtime_id = np.argmax(times)
    max_id = maxtime_id - (1 if times[maxtime_id] == np.inf else 0)
    arclength_diff = np.array([
        spl.arclength(spline, uj, ui)
        for ui, uj in zip(dominoes[:max_id-1], dominoes[1:max_id])])
    time_diff = np.array([
        tj - ti
        for ti, tj in zip(times[:max_id-1], times[1:max_id])])

    if USE_KRV_INSTEAD_OF_ANGLE:
        curvature = np.array([
            spl.curvature(spline, ui) for ui in dominoes[:max_id-1]])
        return arclength_diff, curvature, time_diff

    else:
        angles = spl.splang(dominoes, spline)
        angle_diff = np.array([
            (aj - ai + 180) % 360 - 180
            for ai, aj in zip(angles[:max_id-1], angles[1:max_id])])
        return arclength_diff, angle_diff, time_diff


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    spath = sys.argv[1]
    root, _ = os.path.splitext(spath)
    dpath = root + "-doms.npz"
    d2spath = root + "-dom2spl.npy"
    tpath = root + "-times.npz"
    denpath = root + "-densities.npy"

    with open(spath, 'rb') as f:
        splines = pickle.load(f)
    doms = np.load(dpath)
    dom2spl = np.load(d2spath)
    times = np.load(tpath)
    dens = np.load(denpath)

    # Local variables
    arclength_diffs = []
    angle_diffs = []
    time_diffs = []
    if SHOW_ARCLENGTHS:
        arclengths = []
    # Global variables
    densities = []
    avg_angle_diffs = []
    avg_speeds = []

    for i in range(len(doms.files)):
        arcd, angd, timed = get_local_variables(
            splines[dom2spl[i]],
            doms['arr_{}'.format(i)],
            times['arr_{}'.format(i)])
        # Local variables
        arclength_diffs.extend(arcd)
        angle_diffs.extend(angd)
        time_diffs.extend(timed)
        if SHOW_ARCLENGTHS:
            arclengths.extend(np.cumsum(arcd) / np.sum(arcd))
        # Global variables
        densities.append(dens[i])
        avg_angle_diffs.append(angd.mean())
        avg_speeds.append(timed.sum() / arcd.sum())

    # Remove outliers
    n = 10
    mean, std = np.mean(time_diffs), np.std(time_diffs)
    time_diffs = np.ma.masked_outside(time_diffs, mean - n*std, mean + n*std)
    #  arclength_diffs = np.ma.masked_array(arclength_diffs,
    #                                       mask=time_diffs.mask)
    #  angle_diffs = np.ma.masked_array(angle_diffs, mask=time_diffs.mask)
    mean, std = np.mean(avg_speeds), np.std(avg_speeds)
    avg_speeds = np.ma.masked_outside(avg_speeds, mean - n*std, mean + n*std)
    #  densities = np.ma.masked_array(densities, mask=avg_speeds.mask)
    #  avg_angle_diffs = np.ma.masked_array(avg_angle_diffs,
    #                                       mask=avg_speeds.mask)
    #  if SHOW_ARCLENGTHS:
    #      arclengths = np.ma.masked_less(arclengths, 0.2)
    #      arclength_diffs = np.ma.masked_array(arclength_diffs,
    #                                           mask=arclengths.mask)

    # Figure 1: local
    if COMBINE_PLOTS:
        naxes = SHOW_ANGLES + SHOW_ARCLENGTHS
        fig = plt.figure(1, figsize=(naxes*5, 5))
        axid = 1

        if SHOW_ANGLES:
            ax = fig.add_subplot(1, naxes, axid)
            ax.scatter(arclength_diffs, angle_diffs, c=time_diffs,
                       edgecolor='none')
            ax.set_xlabel("Arclength distance between two dominoes")
            if USE_KRV_INSTEAD_OF_ANGLE:
                ax.set_title("Local toppling time wrt arclength distance "
                             "and curvature")
                ax.set_ylabel("Curvature at the previous domino")
            else:
                ax.set_title("Local toppling time wrt arclength distance "
                             "and angle difference")
                ax.set_ylabel("Angle difference between two dominoes")
            axid += 1

        if SHOW_ARCLENGTHS:
            ax = fig.add_subplot(1, naxes, axid)
            ax.scatter(arclength_diffs, arclengths, c=np.log(time_diffs),
                       edgecolor='none')
            ax.set_xlabel("Arclength distance between two dominoes")
            ax.set_ylabel("Arclength parameter value")
            ax.set_title("Log of local toppling time wrt arclength distance "
                         "and arclength value")

    else:
        naxes = 1 + SHOW_ANGLES + SHOW_ARCLENGTHS
        fig = plt.figure(1, figsize=(naxes*5, 5))
        axid = 1

        plt.set_cmap('flag')
        inds = np.concatenate([np.arange(times['arr_{}'.format(i)].size-2)
                               for i in range(len(times.files))])

        ax = fig.add_subplot(1, naxes, axid)
        ax.scatter(arclength_diffs, time_diffs, c=inds, edgecolor='none')
        ax.set_title("Local toppling time wrt arclength distance")
        ax.set_xlabel("Arclength distance between two dominoes")
        ax.set_ylabel("Local toppling time")
        axid += 1

        if SHOW_ANGLES:
            ax = fig.add_subplot(1, naxes, axid, sharey=ax)
            ax.scatter(angle_diffs, time_diffs, edgecolor='none')
            if USE_KRV_INSTEAD_OF_ANGLE:
                ax.set_title("Local toppling time wrt curvature")
                ax.set_xlabel("Curvature at the previous domino")
            else:
                ax.set_title("Local toppling time wrt angle difference")
                ax.set_xlabel("Angle difference between two dominoes")
            axid += 1

        if SHOW_ARCLENGTHS:
            ax = fig.add_subplot(1, naxes, axid, sharey=ax)
            ax.scatter(arclengths, time_diffs, c=inds, edgecolor='none')
            ax.set_title("Local toppling time wrt arclength")
            ax.set_xlabel("Arclength parameter value")

    dirname = os.path.dirname(spath)
    plt.savefig(dirname+"/local.png", bbox_inches='tight')

    # Figure 2: global
    if COMBINE_PLOTS:
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.scatter(densities, avg_angle_diffs, c=avg_speeds,
                   edgecolor='none')
        ax.set_xlabel("Domino density along the path")
        if USE_KRV_INSTEAD_OF_ANGLE:
            ax.set_title("Total path toppling speed wrt density "
                         "and curvature")
            ax.set_ylabel("Average curvature along the path")
        else:
            ax.set_title("Total path toppling speed wrt density "
                         "and angle difference")
            ax.set_ylabel("Average angle difference between successive\n"
                          "dominoes on the path")
    else:
        naxes = 1 + SHOW_ANGLES
        fig = plt.figure(2, figsize=(naxes*5, 5))
        axid = 1

        ax = fig.add_subplot(1, naxes, axid)
        ax.scatter(densities, avg_speeds, edgecolor='none')
        ax.set_title("Total path toppling speed wrt density")
        ax.set_xlabel("Domino density along the path")
        ax.set_ylabel("Total path toppling speed")
        axid += 1

        if SHOW_ANGLES:
            ax = fig.add_subplot(1, naxes, axid, sharey=ax)
            ax.scatter(avg_angle_diffs, avg_speeds, edgecolor='none')
            if USE_KRV_INSTEAD_OF_ANGLE:
                ax.set_title("Total path toppling speed wrt average curvature")
                ax.set_xlabel("Average curvature along the path")
            else:
                ax.set_title("Total path toppling speed wrt average\n"
                             "angle difference")
                ax.set_xlabel("Average angle difference between successive\n"
                              "dominoes on the path")
    plt.savefig(dirname+"/global.png", bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()
