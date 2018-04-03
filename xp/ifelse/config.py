from math import atan, degrees

from xp.config import *

BALL_RADIUS = 0.015 / 1  # [m]
BALL_MASS = 0.01 / 6  # [kg]
TOP_TRACK_LWH = (0.3, 0.025, 0.005)  # [m]
BOTTOM_TRACK_LWH = TOP_TRACK_LWH  # [m]
HIGH_PLANK_LWH = (0.235, 0.023, 0.008)  # [m]
HIGH_PLANK_MASS = 0.02  # [kg]
LOW_PLANK_LWH = (0.1175, 0.023, 0.016)  # [m]
LOW_PLANK_MASS = 0.02  # [kg]
BASE_PLANK_LWH = (0.35, 0.025, 0.005)  # [m]
BASE_PLANK_MASS = 0.021  # [kg]
ROUND_SUPPORT_RADIUS = 0.014  # [m]
ROUND_SUPPORT_HEIGHT = 0.026  # [m]
FLAT_SUPPORT_LWH = (.02, .025, .005)  # [m]
GOBLET_HEIGHT = 0.119  # [m]
GOBLET_R1 = 0.0455  # [m]
GOBLET_R2 = 0.031  # [m]

SCENARIO_PARAMETERS_BOUNDS = (
    (-BASE_PLANK_LWH[0]/4, BASE_PLANK_LWH[0]/2),
    (HIGH_PLANK_LWH[0]/2, HIGH_PLANK_LWH[0]+TOP_TRACK_LWH[0]),
    (-75, -5),
    (LOW_PLANK_LWH[0]/2, HIGH_PLANK_LWH[0]),
    (5, 45),
    (BASE_PLANK_LWH[0]/2 + GOBLET_R1/2, BASE_PLANK_LWH[0]),
    (-LOW_PLANK_LWH[0], LOW_PLANK_LWH[0]),
    (-60, 0),
)

PIVOTING_ANGULAR_VELOCITY = 1
ROLLING_ANGLE = 90
HIGH_PLANK_TOPPLING_ANGLE = degrees(atan(HIGH_PLANK_LWH[2]/HIGH_PLANK_LWH[0]))
STOPPING_LINEAR_VELOCITY = 1e-2
STOPPING_ANGULAR_VELOCITY = 1
