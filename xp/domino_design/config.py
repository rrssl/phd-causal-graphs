import sys
import os

sys.path.insert(0, os.path.abspath("../.."))
from xp.config import *


# Path to the classifier.
SVC_PATH = "/media/DATA/pro/2_Contraptions/data/xp/domino_learning/20170927/samples-3D-classifier.pkl"
SVC2_PATH = "/media/DATA/pro/2_Contraptions/data/xp/domino_learning/20170913-3/samples-3D-sym-classifier.pkl"

# Spline interpolation parameters.

# PATH_SIZE_RATIO := area(path's bounding box) / area(domino's smallest face)
PATH_SIZE_RATIO = 25
SMOOTHING_FACTOR = .1
# Variant: use with "EG"
#  PATH_SIZE_RATIO = 80
#  SMOOTHING_FACTOR = .02
# Variant: challenging case with 20170812-182213.npy (the "W")
#  PATH_SIZE_RATIO = 15
#  SMOOTHING_FACTOR = .01

# Ranges for random sampling of the previous parameters.
MIN_SIZE_RATIO = 5
MAX_SIZE_RATIO = 100
MIN_SMOOTHING_FACTOR = .001
MAX_SMOOTHING_FACTOR = .5

# Number of trials for each run with a given level of uncertainty.
NTRIALS_UNCERTAINTY = 10
