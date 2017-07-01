#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determining variances in the sytem.

@author: Robin Roussel
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from panda3d.bullet import BulletWorld
from panda3d.core import NodePath
from panda3d.core import Vec3

from primitives import DominoMaker, Floor


global_params = {
        'extents': Vec3(.05, .2, .4),
        'mass': 1.
    }
toppling_angle = math.atan(
        global_params['extents'][0] / global_params['extents'][2])
print(toppling_angle)

# Meta parameters
angvel_init = Vec3(0., 5., 0.)
timestep = 1. / 60.
time_per_domino = .5
nb_trials = 10
DEBUG_SHOW = 0


def run_simu(params, timestep):
    """Run the simulation for that specific example."""
    # World
    world = BulletWorld()
    world.set_gravity(Vec3(0, 0, -9.81))
    # Floor
    floor_path = NodePath("floor")
    floor = Floor(floor_path, world)
    floor.create()
    # Dominos
    dom_path = NodePath("dominos")
    dom_fact = DominoMaker(dom_path, world)
    for i, (p, h) in enumerate(zip(params['pos'], params['head'])):
        dom_fact.add_domino(
                Vec3(*p), h, prefix="domino_{}".format(i), **global_params)

    first_dom = dom_path.get_child(0)
    first_dom.node().set_angular_velocity(angvel_init)

    if DEBUG_SHOW:
        from viewers import Modeler
        m = Modeler()
        dom_path.reparent_to(m.models)
        m.run()

    # Run the simulation.
    t = 0.
    while t < time_per_domino * dom_path.get_num_children():
        t += timestep
        world.do_physics(timestep)

    return dom_path


def toppled(domino):
    """Evaluate whether a domino toppled or not."""
    # print(abs(domino.get_p()) * math.pi / 180.)
    return abs(domino.get_r()) * math.pi / 180. > toppling_angle


def run_xp1():
    print("XP1: evaluating simulator variance")
    nb_dominos = np.arange(3, 9)
    print("Successive numbers of dominos: {}".format(nb_dominos))
    step = .06 + np.arange(6) * .1
    print("Successive steps: {}".format(step))

    probas = np.zeros((len(nb_dominos), len(step)))
    for i, n in enumerate(nb_dominos):
        for j, s in enumerate(step):
            success = []
            for k in range(nb_trials):
                params = {
                        'pos': np.column_stack([
                            np.arange(n) * s,
                            np.zeros(n),
                            np.full(n, global_params['extents'][2] / 2)
                            ]),
                        'head': np.zeros(n)
                        }
                dom_path = run_simu(params, timestep)
                success.append(all(toppled(d) for d in dom_path.get_children()))
                # if k == 0:
                    # print([toppled(d) for d in dom_path.get_children()])
            probas[i, j] = success.count(True) / nb_trials
    print("Probabilities of success:\n", probas)

    # Process results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(probas, cmap=plt.cm.viridis,
                    interpolation='nearest')

    width, height = probas.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(probas[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res, ticks=[0., 1.])
    plt.xticks(range(height), step)
    plt.yticks(range(width), nb_dominos[::-1])
    ax.set_xticks([.5, 1.5, 2.5, 3.5, 4.5], minor=True);
    ax.set_yticks([.5, 1.5, 2.5, 3.5, 4.5], minor=True);
    ax.tick_params(which='both', length=0)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.set_xlabel("Space between dominoes")
    ax.set_ylabel("Number of dominoes")
    plt.show()


def run_xp2():
    print("XP2: measuring probability of success wrt a single source of "
          "error.")
    nb_dominos = 10
    step = .25
    print("Step = {}; Number of dominos = {}.".format(step, nb_dominos))
    alpha = 5 + np.arange(9) * 5
    print("Successive values of the symmetric uniform distribution "
          "parameter (in degrees): {}".format(alpha))

    probas = np.zeros(len(alpha))
    for i, a in enumerate(alpha):
        success = []
        for _ in range(nb_trials):
            params = {
                    'pos': np.column_stack([
                        np.arange(nb_dominos) * step,
                        np.zeros(nb_dominos),
                        np.full(nb_dominos, global_params['extents'][2] / 2)
                        ]),
                    'head': 2 * a * np.random.random(nb_dominos) - a
                    }
            dom_path = run_simu(params, timestep)
            success.append(all(toppled(d) for d in dom_path.get_children()))
        probas[i] = success.count(True) / nb_trials
    print("Probabilities of success:\n", probas)

    # Process results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(alpha, probas)
    ax.set_xlabel("Symmetric uniform distribution parameter (in degrees)")
    ax.set_ylabel("Probability of success")
    plt.show()


def run_xp3():
    print("XP3: measuring the correlation between probability decrease "
          "and control parameters.")
    nb_dominos = 10
    print("Number of dominos = {}.".format(nb_dominos))
    step = .06 + np.arange(6) * .05
    print("Successive steps: {}".format(step))
    alpha = 5 + np.arange(9) * 5
    print("Successive values of the symmetric uniform distribution "
          "parameter (in degrees): {}".format(alpha))

    probas = np.zeros((len(step), len(alpha)))
    for i, s in enumerate(step):
        for k, a in enumerate(alpha):
            success = []
            for _ in range(nb_trials):
                params = {
                        'pos': np.column_stack([
                            np.arange(nb_dominos) * s,
                            np.zeros(nb_dominos),
                            np.full(nb_dominos,
                                    global_params['extents'][2] / 2)
                            ]),
                        'head': 2 * a * np.random.random(nb_dominos) - a
                        }
                dom_path = run_simu(params, timestep)
                success.append(all(toppled(d)
                               for d in dom_path.get_children()))
            probas[i, k] = success.count(True) / nb_trials
    print("Probabilities of success:\n", probas)

    # Process results
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    for i in range(len(step)):
        # ax = fig.add_subplot(2, 3, i+1)
        axes.flat[i].scatter(alpha, probas[i])
        axes.flat[i].set_title("Step = {:.2f}".format(step[i]))
        # ax.set_xlabel("Symmetric uniform distribution parameter (in degrees)")
        # ax.set_ylabel("Probability of success")
    plt.show()


run_xp1()
# nb_trials = 30
# run_xp2()
# run_xp3()
