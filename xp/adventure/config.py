from math import atan, degrees

import xp.config
from xp.config import GRAVITY  # noqa: F401

BALL_RADIUS = 0.01  # [m]
BALL_MASS = 0.01  # [kg] TODO
BALL_FRICTION = .1
TOP_TRACK_LWHT = (0.35, 0.025, 0.005, .002)  # [m]
TOP_TRACK_FRICTION = 1
TINY_TRACK_LWH = (0.1, 0.025, 0.005)  # [m]
TINY_TRACK_MASS = .01  # [kg] TODO
SHORT_TRACK_LWHT = (0.15, 0.025, 0.008, .002)  # [m]
LONG_TRACK_LWHT = (0.3, 0.025, 0.008, .002)  # [m]
FLAT_SUPPORT_LWH = (.02, .025, .005)  # [m]
GOBLET_HEIGHT = 0.11  # [m]
GOBLET_R1 = 0.036  # [m]
GOBLET_R2 = 0.025  # [m]
GOBLET_MASS = .003  # [g]
GOBLET_FRICTION = 1
GOBLET_ANGULAR_DAMPING = 1
NAIL_LEVER_LWH = (0.11, 0.005, 0.002)  # [m]
NAIL_LEVER_MASS = 0.005  # [kg]
PLANK_LWH = (0.1175, 0.023, 0.008)  # [m]
PLANK_MASS = 0.01  # [kg]
QUAD_PLANK_LWH = (0.1175, 0.023, 0.031)  # [m]
RIGHT_WEIGHT_HEIGHT = 0.0315  # [m]
RIGHT_WEIGHT_RADIUS = 0.0315 / 2  # [m]
RIGHT_WEIGHT_MASS = 0.2  # [kg]
TEAPOT_LID_RADIUS = GOBLET_R1
TEAPOT_LID_HEIGHT = .005  # [m]
TEAPOT_LID_MASS = 0.005  # [kg] TODO
TEAPOT_FRICTION = .1
TOP_PULLEY_ROPE_LENGTH = .57  # [m]
LEFT_PULLEY_ROPE_LENGTH = .95  # [m]
RIGHT_PULLEY_ROPE_LENGTH = 1.  # [m]
RIGHT_PULLEY_PIVOT_RADIUS = .006  # [m]
RIGHT_PULLEY_PIVOT_HEIGHT = .003  # [m]
RIGHT_PULLEY_PIVOT_COILED = .009  # [m]

SCENARIO_PARAMETERS_BOUNDS = (
    (-.20, -.10),
    (.05, .20),
    (1., 30.),
    (-30., -1.),
    (-.40, -.20),
    (-.30, -.10),
    (1., 30.),
    (-.30, -.10),
    (-.35, -.15),
    (-30., -1.),
    (-.30, -.10),
    (-.40, -.20),
    (1., 30.),
    (1., 30.),
    (.20, .40),
    (-.20, -.10),
    (-10., -1.),
    (.15, .35),
    (-.25, -.15),
    (1., 10.),
    (.20, .40),
    (-.35, -.20),
    (-30., -1.),
    (-.45, -.15),
    (-.30, -.10),
    (.15, .45),
    (-.25, -.05),
    (-.10, 0.),
    (.05, .20),
    (-.10, 0.),
    (-.40, -.20),
    (-.10, 0.),
    (-.55, -.45),
    (.05, .20),
    (.25, .40),
    (-.10, 0.),
    (-.05, .10),
    (.40, .60),
)

xp.config.MAX_WAIT_TIME = 4
PIVOTING_ANGULAR_VELOCITY = 1
ROLLING_ANGLE = 90
RISING_LINEAR_VELOCITY = 1e-2
FALLING_LINEAR_VELOCITY = -1e-2
HIGH_PLANK_TOPPLING_ANGLE = degrees(atan(0))
STOPPING_LINEAR_VELOCITY = 5e-2
STOPPING_ANGULAR_VELOCITY = 10
