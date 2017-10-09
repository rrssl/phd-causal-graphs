import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from xp.config import *

TIMESTEP = 1/60
TIMESTEP_PRECISE = timestep

MIN_DENSITY = .2
MAX_DENSITY = .5
MIN_SPACING = .008
MAX_SPACING = .035

NCORES = 7
