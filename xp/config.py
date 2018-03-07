"""
Global parameters.

"""
from math import atan, degrees
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from config import *


# Number of cores to use for parallelization.
NCORES = 7

# Whether to reduce symmetries when learning robustness.
REDUCE_SYM = True

# Simulation parameters.
TIMESTEP = 1/500            # [s]
MAX_WAIT_TIME = 2.          # [s]

# Floor properties.
FLOOR_MATERIAL_FRICTION = 0.50447012
FLOOR_MATERIAL_RESTITUTION = 0.

# Domino properties.
h = .044
w = .02
t = .007
DOMINO_EXTENTS = (t, w, h)
MASS = .00305  # Legacy.
DOMINO_MASS = MASS
DOMINO_MATERIAL_FRICTION = 0.62309873
DOMINO_MATERIAL_RESTITUTION = 0.75442644
DOMINO_ANGULAR_DAMPING = 0.08735053
# Computed values
TOPPLING_ANGLE = degrees(atan(t / h))     # [degrees]

# Plank properties.
PLANK_LENGTH = .1       # [m]
PLANK_WIDTH = .04       # [m]
PLANK_THICKNESS = .005   # [m]
PLANK_EXTENTS = (PLANK_LENGTH, PLANK_WIDTH, PLANK_THICKNESS)
PLANK_MATERIAL_FRICTION = 0.5
PLANK_MATERIAL_RESTITUTION = 0.

# Ball properties.
BALL_RADIUS = .015/2    # [m]
BALL_MASS = .01/6       # [kg]
BALL_MATERIAL_FRICTION = .5
BALL_MATERIAL_RESTITUTION = 0.
BALL_LINEAR_DAMPING = 0.
BALL_ANGULAR_DAMPING = 0.

# Bounds of the configuration space.
X_MIN = t               # min. x in the relative frame of reference [m]
X_MAX = 1.2 * h         # max. x in the relative frame of reference [m]
Y_MIN = -w * 1.1        # min. y in the relative frame of reference [m]
Y_MAX = w * 1.1         # max. y in the relative frame of reference [m]
A_MIN = -75.            # min. relative orientation [degrees]
A_MAX = 75.             # max. relative orientation [degrees]
MIN_SPACING = .008      # min. spacing between the dominoes [m]
MAX_SPACING = .035      # max. spacing between the dominoes [m]
PLANK_X_MIN = -PLANK_LENGTH         # in the frame of the domino [m]
PLANK_X_MAX = 0.                    # in the frame of the domino [m]
PLANK_Y_MIN = 0                     # in the frame of the domino [m]
PLANK_Y_MAX = PLANK_LENGTH          # in the frame of the domino [m]
PLANK_A_MIN = -75.      # min. angle of the plank wrt the ground [degrees]
PLANK_A_MAX = 0.        # max. angle of the plank wrt the ground [degrees]
