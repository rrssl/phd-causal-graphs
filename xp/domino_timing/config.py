import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from predicting_domino_toppling.config import t, w, h, density, timestep, MAX_WAIT_TIME


NCORES = 7
MIN_DENSITY = .2
MAX_DENSITY = .5
