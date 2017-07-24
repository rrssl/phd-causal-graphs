"""
Processing domino-pair samples. For each sample, test whether if the first
domino is just out of equilibrium, the second topples.

"""
import numpy as np

from functions import run_domino_toppling_xp
from config import timestep, maxtime, density


def process(samples):
    values = np.empty(len(samples))
    #  for i, (d, a, h, t) in enumerate(samples):
    h = .05
    w = h / 3.
    t = h / 10.
    m = density * t * w * h
    for i, (d, a) in enumerate(samples):
        values[i] = run_domino_toppling_xp(
                (t, w, h, d, a, m), timestep, maxtime)

    return values


def main():
    samples = np.load("samples-2D.npy")
    values = process(samples)
    np.save("values-2D.npy", values)
    #  print(values)

main()
