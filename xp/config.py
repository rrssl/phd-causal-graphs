"""
Global parameters.

"""
from math import atan, pi


## Number of cores to use for parallelization.
NCORES = 7

## Domino properties.
# Arbitrary values
#  h = .05                     # height [m]
#  w = h / 3.                  # width [m]
#  t = h / 10.                 # thickness [m]
#  DENSITY = 650.              # [kg/m^3]
#  MASS = t * w *h * DENSITY   # [kg]
# Experimental values
h = .044
w = .02
t = .007
MASS = .00305
# Computed values
TOPPLING_ANGLE = atan(t / h) * 180 / pi     # [degrees]

## Simulation parameters.
TIMESTEP = 1/500            # [s]
MAX_WAIT_TIME = 2.          # [s]

## Bounds of the configuration space.
X_MIN = 0               # min. x in the relative frame of reference [m]
X_MAX = 1.5 * h         # max. x in the relative frame of reference [m]
Y_MIN = -w * 1.1        # min. y in the relative frame of reference [m]
Y_MAX = w * 1.1         # max. y in the relative frame of reference [m]
A_MIN = -90             # min. relative orientation [degrees]
A_MAX = 90              # max. relative orientation [degrees]
MIN_SPACING = .008      # min. spacing between the previous dominoes [m]
MAX_SPACING = .035      # max. spacing between the previous dominoes [m]
