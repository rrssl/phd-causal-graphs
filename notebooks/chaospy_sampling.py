# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3 (jupyter)
#     language: python
#     name: python3
# ---

# +
import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np

mu = 1
sigma = 2

full_gaussian = cp.Normal(mu, sigma)
truncated_gaussian = cp.Truncnorm(mu, 3, mu, sigma)
folded_gaussian = cp.Foldnormal(mu, sigma, mu)

x = np.linspace(mu-5*sigma, mu+5*sigma, 100)
fig, ax = plt.subplots()
ax.plot(x, full_gaussian.pdf(x))
ax.plot(x, truncated_gaussian.pdf(x))
ax.plot(x, folded_gaussian.pdf(x), '--')

# +
ns = 1000

samples_g = full_gaussian.sample(ns, rule='H')
samples_tg = truncated_gaussian.sample(ns, rule='H')
# samples_fg = folded_gaussian.sample(ns, rule='R')  # This currently raises an error

fig, ax = plt.subplots()
ax.hist(samples_g, alpha=.8, bins='auto')
ax.hist(samples_tg, alpha=.8, bins='auto')
# -


