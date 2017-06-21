#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Determining variances in the sytem.

@author: Robin Roussel
"""
import math
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

    # Run the simulation.
    t = 0.
    while t < time_per_domino * dom_path.get_num_children():
        t += timestep
        world.do_physics(timestep)

    if DEBUG_SHOW:
        from viewers import Modeler
        m = Modeler()
        dom_path.reparent_to(m.models)
        m.run()

    return dom_path


def toppled(domino):
    """Evaluate whether a domino toppled or not."""
    # print(abs(domino.get_p()) * math.pi / 180.)
    return abs(domino.get_r()) * math.pi / 180. > toppling_angle

# Experiment 1
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


# Experiment 2
# Experiment 3
