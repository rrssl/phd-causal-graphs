"""
Sampling the parameter space. Enter n as the number of samples.

Parameters:
    d: distance between dominoes
    a: heading angle
    h: height
    t: thickness
"""
from math import pi, atan
import numpy as np
from random import uniform
import sys

from functions import make_box, tilt_box_forward, has_contact


def sample(n):
    samples = np.empty((n, 2))
    i = 0
    while i < n:
        #  h = uniform(.01, .1)
        h = .05
        a = uniform(0., 90.)
        #  t = uniform(.001, h / 3.)
        t = h / 10.
        d = uniform(t, 1.5 * h)

        w = h / 3.
        d1 = make_box((t, w, h), (0, 0, h*.5), (0, 0, 0))
        d2 = make_box((t, w, h), (d, 0, h*.5), (a, 0, 0))
        tilt_box_forward(d1, atan(t / h) * 180 / pi + 1.)
        if has_contact(d1, d2):
            pass
        else:
            #  samples[i] = d, a, h, t
            samples[i] = d, a
            i += 1

    return samples


def main():
    argv = sys.argv
    if len(argv) <= 1:
        print("No input n, using n = 10.")
        n = 10
    else:
        n = int(argv[1])
    s = sample(n)
    np.save("samples-2D.npy", s)


main()
