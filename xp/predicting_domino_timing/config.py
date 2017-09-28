import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from predicting_domino_toppling.config import t, w, h, density
from predicting_domino_toppling.config import timestep, MAX_WAIT_TIME
from predicting_domino_toppling.config import X_MIN, X_MAX, Y_MIN, Y_MAX, A_MIN, A_MAX


MIN_DENSITY = .2
MAX_DENSITY = .5
MIN_SPACING = .008
MAX_SPACING = .035

NCORES = 7
