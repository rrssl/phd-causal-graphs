import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from xp.config import *

TIMESTEP_PRECISE = TIMESTEP
TIMESTEP_FAST = 1/60

MIN_DENSITY = .2
MAX_DENSITY = .5
