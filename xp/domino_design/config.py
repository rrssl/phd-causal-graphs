import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from domino_learning.config import t, w, h


SVC_PATH = "../domino_learning/samples-3D-model.pkl"
MIN_SIZE_RATIO = 5
MAX_SIZE_RATIO = 50
MIN_SMOOTHING_FACTOR = .001
MAX_SMOOTHING_FACTOR = .5
