import sys
import os

sys.path.insert(0, os.path.abspath(".."))
from predicting_domino_toppling.config import t, w, h
from predicting_domino_toppling.config import density
from predicting_domino_toppling.config import X_MIN, X_MAX
from predicting_domino_toppling.config import Y_MIN, Y_MAX
from predicting_domino_toppling.config import A_MIN, A_MAX
from predicting_domino_toppling.config import MAX_WAIT_TIME


# Number of cores to use for parallelization.
NCORES = 4

# Path to the classifier.
SVC_PATH = "../../../data/xp/predicting_domino_toppling/20170824/samples-3D-model.pkl"
SVC2_PATH = "../../../data/xp/predicting_domino_toppling/20170913-3/samples-3D-sym-classifier.pkl"

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
