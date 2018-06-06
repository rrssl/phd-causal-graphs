import xp.config
from xp.config import GRAVITY, NCORES  # noqa: F401

BALL_RADIUS = 0.01  # [m]
BALL_MASS = 0.02  # [kg] TODO
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
TEAPOT_LID_MASS = 0.002  # [kg] TODO
TEAPOT_LID_ANGULAR_DAMPING = 1
TEAPOT_FRICTION = .1
TOP_PULLEY_ROPE_LENGTH = .57  # [m]
LEFT_PULLEY_ROPE_LENGTH = .95  # [m]
RIGHT_PULLEY_ROPE_LENGTH = 1.  # [m]
RIGHT_PULLEY_PIVOT_RADIUS = .006  # [m]
RIGHT_PULLEY_PIVOT_HEIGHT = .003  # [m]
RIGHT_PULLEY_PIVOT_COILED = .009  # [m]

PARAMETER_LABELS = [
    "top track x",
    "top track y",
    "top track a",
    "left track 1 a",
    "left track 2 x",
    "left track 2 y",
    "left track 2 a",
    "left track 3 x",
    "left track 3 y",
    "left track 3 a",
    "left track 4 x",
    "left track 4 y",
    "left track 4 a",
    "right track 1 a",
    "right track 2 x",
    "right track 2 y",
    "right track 2 a",
    "right track 3 x",
    "right track 3 y",
    "right track 3 a",
    "right track 4 x",
    "right track 4 y",
    "right track 4 a",
    "left pulley weight x",
    "left pulley weight y",
    "right pulley weight x",
    "right pulley weight y",
    "gate x",
    "bridge x",
    "bridge y",
    "bottom goblet x",
    "bottom goblet and teapot y",
    "teapot x",
    "top pulley p1 & p2 y",
    "left pulley p1 & p2 y",
    "right pulley p1 y",
    "right pulley p2 x"
]

MANUAL_SCENARIO_PARAMETERS = [
    -.30,     # top track x
    .11,      # top track y
    11,       # top track a
    -11,      # left track 1 a
    -.32,     # left track 2 x
    -.11,     # left track 2 y
    10,       # left track 2 a
    -.22,     # left track 3 x
    -.16,     # left track 3 y
    -22,      # left track 3 a
    -.24,     # left track 4 x
    -.25,     # left track 4 y
    15,       # left track 4 a
    15,       # right track 1 a
    .33,      # right track 2 x
    -.12,     # right track 2 y
    -2,       # right track 2 a
    .25,      # right track 3 x
    -.18,     # right track 3 y
    2,        # right track 3 a
    .32,      # right track 4 x
    -.27,     # right track 4 y
    -20,      # right track 4 a
    -.411,    # left pulley weight x
    -.11,     # left pulley weight y
    .43,      # right pulley weight x
    -.10,     # right pulley weight y
    -.07,     # gate x
    -.05,     # bridge x
    -.30,     # bridge y
    -.05,     # bottom goblet x
    -.50,     # bottom goblet and teapot y
    .13,      # teapot x
    .30,      # top pulley p1 & p2 y
    -.04,     # left pulley p1 & p2 y
    0.,       # right pulley p1 y
    .50       # right pulley p2 x
]

SCENARIO_PARAMETERS_INTERVALS = [
    (-.03, .03),    # top track x
    (-.03, .03),    # top track y
    (-5, 5),        # top track a
    (-5, 5),        # left track 1 a
    (-.03, .03),    # left track 2 x
    (-.03, .03),    # left track 2 y
    (-5, 5),        # left track 2 a
    (-.03, .03),    # left track 3 x
    (-.03, .03),    # left track 3 y
    (-5, 5),        # left track 3 a
    (-.03, .03),    # left track 4 x
    (-.03, .03),    # left track 4 y
    (-5, 5),        # left track 4 a
    (-5, 5),        # right track 1 a
    (-.03, .03),    # right track 2 x
    (-.03, .03),    # right track 2 y
    (-5, 5),        # right track 2 a
    (-.03, .03),    # right track 3 x
    (-.03, .03),    # right track 3 y
    (-5, 5),        # right track 3 a
    (-.03, .03),    # right track 4 x
    (-.03, .03),    # right track 4 y
    (-5, 5),        # right track 4 a
    (-.03, .03),    # left pulley weight x
    (-.03, .03),    # left pulley weight y
    (-.03, .03),    # right pulley weight x
    (-.03, .03),    # right pulley weight y
    (-.03, .03),    # gate x
    (-.03, .03),    # bridge x
    (-.03, .03),    # bridge y
    (-.03, .03),    # bottom goblet x
    (-.03, .03),    # bottom goblet and teapot y
    (-.03, .03),    # teapot x
    (-.03, .03),    # top pulley p1 & p2 y
    (-.03, .03),    # left pulley p1 & p2 y
    (-.03, .03),    # right pulley p1 y
    (-.03, .03),    # right pulley p2 x
]
SCENARIO_PARAMETERS_BOUNDS = [
    (x + lower, x + upper) for x, (lower, upper)
    in zip(MANUAL_SCENARIO_PARAMETERS, SCENARIO_PARAMETERS_INTERVALS)
]

xp.config.MAX_WAIT_TIME = 4
PIVOTING_ANGULAR_VELOCITY = 1
ROLLING_ANGLE = 90
RISING_LINEAR_VELOCITY = 1e-2
FALLING_LINEAR_VELOCITY = -1e-2
STOPPING_LINEAR_VELOCITY = 5e-2
STOPPING_ANGULAR_VELOCITY = 20

SOFTMIN_COEFF = .8
