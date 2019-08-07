from .ball import Ball
from .base import World
from .box import Box
from .cylinder import Cylinder
from .domino_run import DominoRun
from .empty import Empty
from .fastener import Fastener
from .goblet import Goblet
from .lever import Lever
from .open_box import OpenBox
from .pivot import Pivot
from .plane import Plane
from .pulley import Pulley
from .rope_pulley import RopePulley
from .tension_rope import TensionRope
from .track import Track


def get_primitives():
    return (Plane, Ball, Box, Cylinder, Lever, Pulley, Goblet, DominoRun,
            TensionRope, RopePulley, Track, OpenBox)
