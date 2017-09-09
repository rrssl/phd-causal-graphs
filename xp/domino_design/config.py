import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from domino_learning.config import t, w, h
from domino_learning.config import density
from domino_learning.config import X_MIN, X_MAX
from domino_learning.config import Y_MIN, Y_MAX
from domino_learning.config import A_MIN, A_MAX


# Number of cores to use for parallelization.
NCORES = 4

# Path to the classifier.
SVC_PATH = "../../../data/xp/domino_learning/samples-3D-model.pkl"

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
