import core.primitives as prim

BALL_FRICTION = .1
# BALL_MASS = 0.013  # [kg]
BALL_MASS = 0.0056  # [kg]
# BALL_RADIUS = 0.0105  # [m]
BALL_RADIUS = 0.008  # [m]
BALL_RESTITUTION = 0.8
FLAT_SUPPORT_LWH = (.02, .025, .005)  # [m]
GOBLET_ANGULAR_DAMPING = 1
GOBLET_FRICTION = 1
GOBLET_HEIGHT = 0.11  # [m]
GOBLET_MASS = .003  # [g]
GOBLET_R1 = 0.036  # [m]
GOBLET_R2 = 0.025  # [m]
LEFT_PULLEY_ROPE_LENGTH = .95  # [m]
LONG_TRACK_LWHT = (0.3, 0.025, 0.008, .002)  # [m]
NAIL_LEVER_LWH = (0.11, 0.005, 0.002)  # [m]
NAIL_LEVER_MASS = 0.005  # [kg]
PLANK_LWH = (0.1175, 0.023, 0.008)  # [m]
PLANK_MASS = 0.01  # [kg]
QUAD_PLANK_LWH = (0.1175, 0.023, 0.031)  # [m]
RIGHT_PULLEY_PIVOT_COILED = .009  # [m]
RIGHT_PULLEY_PIVOT_HEIGHT = .003  # [m]
RIGHT_PULLEY_PIVOT_RADIUS = .006  # [m]
RIGHT_PULLEY_ROPE_LENGTH = 1.  # [m]
RIGHT_WEIGHT_HEIGHT = 0.0315  # [m]
RIGHT_WEIGHT_MASS = 0.2  # [kg]
RIGHT_WEIGHT_RADIUS = 0.0315 / 2  # [m]
SHORT_TRACK_LWHT = (0.15, 0.025, 0.008, .002)  # [m]
TEAPOT_FRICTION = .1
TEAPOT_LID_ANGULAR_DAMPING = 1
TEAPOT_LID_HEIGHT = .005  # [m]
TEAPOT_LID_MASS = 0.002  # [kg] TODO
TEAPOT_LID_RADIUS = GOBLET_R1
TINY_TRACK_LWH = (0.1, 0.025, 0.005)  # [m]
TINY_TRACK_MASS = .01  # [kg] TODO
TOP_PULLEY_ROPE_LENGTH = .57  # [m]
TOP_TRACK_FRICTION = 1
TOP_TRACK_LWHT = (0.35, 0.025, 0.005, .002)  # [m]

PIVOTING_ANGULAR_VELOCITY = 1
ANGVEL = 10
STOPPING_LINEAR_VELOCITY = 1e-2
STOPPING_ANGULAR_VELOCITY = 1

middle_track = prim.Track("middle_track", SHORT_TRACK_LWHT,)
left_track1 = prim.Track("left_track1", LONG_TRACK_LWHT,)
left_track2 = prim.Track("left_track2", SHORT_TRACK_LWHT,)
left_track3 = prim.Track("left_track3", SHORT_TRACK_LWHT,)
left_track3_blocker = prim.Box("left_track3_blocker", FLAT_SUPPORT_LWH,)
left_track4 = prim.Track("left_track4", LONG_TRACK_LWHT,)
left_track4_blocker = prim.Box("left_track4_blocker", FLAT_SUPPORT_LWH,)
right_track1 = prim.Track("right_track1", LONG_TRACK_LWHT,)
right_track2 = prim.Track("right_track2", SHORT_TRACK_LWHT,)
right_track2_blocker1 = prim.Box("right_track2_blocker1", FLAT_SUPPORT_LWH,)
right_track2_blocker2 = prim.Box("right_track2_blocker2", FLAT_SUPPORT_LWH,)
right_track3 = prim.Track("right_track3", SHORT_TRACK_LWHT,)
right_track3_blocker = prim.Box("right_track3_blocker", FLAT_SUPPORT_LWH,)
right_track4 = prim.Track("right_track4", SHORT_TRACK_LWHT,)
right_track4_blocker = prim.Box("right_track4_blocker", FLAT_SUPPORT_LWH,)
top_weight_support = prim.Box("top_weight_support", TINY_TRACK_LWH,)
top_weight_guide = prim.Box("top_weight_guide", FLAT_SUPPORT_LWH,)
left_weight_support = prim.Box("left_weight_support", FLAT_SUPPORT_LWH,)
right_weight_support = prim.Box("right_weight_support", FLAT_SUPPORT_LWH,)
top_goblet = prim.Goblet("top_goblet", (GOBLET_HEIGHT, GOBLET_R1, GOBLET_R2),
                         mass=GOBLET_MASS, friction=GOBLET_FRICTION,
                         angular_damping=GOBLET_ANGULAR_DAMPING)
bottom_goblet = prim.Goblet("bottom_goblet", (GOBLET_HEIGHT, GOBLET_R1,
                                              GOBLET_R2),)
sliding_plank = prim.Box("sliding_plank", PLANK_LWH, mass=PLANK_MASS)
gate = prim.Box("gate", PLANK_LWH, mass=PLANK_MASS)
left_weight = prim.Box("left_weight", QUAD_PLANK_LWH, mass=4*PLANK_MASS,)
right_weight = prim.Cylinder("right_weight", (RIGHT_WEIGHT_RADIUS,
                                              RIGHT_WEIGHT_HEIGHT),
                             mass=RIGHT_WEIGHT_MASS)
bridge = prim.Box("bridge", TINY_TRACK_LWH, mass=TINY_TRACK_MASS)
teapot_base = prim.Goblet("teapot_base", (GOBLET_HEIGHT, GOBLET_R1, GOBLET_R2),
                          friction=TEAPOT_FRICTION)
teapot_lid = prim.Cylinder("teapot_lid", (TEAPOT_LID_RADIUS,
                                          TEAPOT_LID_HEIGHT),
                           mass=TEAPOT_LID_MASS,
                           angular_damping=TEAPOT_LID_ANGULAR_DAMPING)
nail = prim.Box("nail", NAIL_LEVER_LWH, mass=NAIL_LEVER_MASS)
nail_pivot = prim.Pivot("pivot", nail,
                        (-NAIL_LEVER_LWH[0]*.2, 0, 0),  # magical
                        (0, 90, 0),)
top_pulley = prim.RopePulley("top_pulley", top_goblet, gate, (0),
                             (-PLANK_LWH[0]/2, 0, 0), TOP_PULLEY_ROPE_LENGTH,)
left_pulley = prim.RopePulley("left_pulley", left_weight, teapot_lid,
                              (-PLANK_LWH[0]/2, 0, 0),
                              (0, 0, TEAPOT_LID_HEIGHT)/2,
                              LEFT_PULLEY_ROPE_LENGTH,)

DATA = {
    'scene': [
        {
            'name': "",
            'type': "",
            'args': {
                '': 0,
            }
        },
        {
            'name': "",
            'type': "",
            'args': {
                '': 0,
            },
            'parent': "",
            'xform': {
                'value': [],
                'range': [
                    [],
                    None,
                    [],
                    None,
                    None,
                    []
                ]
            }
        },
        {
            'name': "ball1",
            'type': "Ball",
            'args': {
                'radius': BALL_RADIUS,
                'b_mass': BALL_MASS,
                'b_friction': BALL_FRICTION,
                'b_restitution': BALL_RESTITUTION
            },
            'parent': "",
            'xform': {
                'value': [0, 0, BALL_RADIUS+.002, 0, 0, 0]
            }
        },
        {
            'name': "ball2",
            'type': "Ball",
            'args': {
                'radius': BALL_RADIUS,
                'b_mass': BALL_MASS,
                'b_restitution': BALL_RESTITUTION
            },
            'parent': "",
            'xform': {
                'value': [0, 0, BALL_RADIUS+.002, 0, 0, 0]
            }
        },
        {
            'name': "",
            'type': "Track",
            'args': {
                'extents': [],
            },
            'xform': {
                'value': [],
                'range': [
                    [],
                    None,
                    [],
                    None,
                    None,
                    []
                ]
            }
        },
        {
            'name': "goblet",
            'type': "Goblet",
            'args': {
                'extents': [
                    GOBLET_HEIGHT,
                    GOBLET_R1,
                    GOBLET_R2
                ]
            },
            'xform': {
                'value': [-.35, GOBLET_R1, -.08, 0, 0, 45],
                'range': [
                    [],
                    None,
                    [],
                    None,
                    None,
                    []
                ]
            }
        },
    ],
    'causal_graph': [
        {
            'name': "",
            'type': "RollingOn",
            'args': {
                'rolling': "",
                'support': "",
                'min_angvel': 0
            },
            'children': [
                "",
            ]
        },
        {
            'name': "",
            'type': "Contact",
            'args': {
                'first': "",
                'second': ""
            },
            'children': [
                "",
                "",
            ]
        },
        {
            'name': "",
            'type': "",
            'args': {
                'body': "",
                'angle': 0,
            },
            'children': [
                "",
            ]
        },
        {
            'name': "",
            'type': "Pivoting",
            'args': {
                'body': "",
                'min_angvel': 0,
            },
            'children': [
                "",
            ]
        },
        {
            'name': "",
            'type': "NoContact",
            'args': {
                'first': "",
                'second': "",
            },
            'children': [
                "",
            ]
        },
        {
            'name': "",
            'type': "Inclusion",
            'args': {
                'inside': "",
                'outside': "goblet",
            },
            'children': [
                "",
            ]
        },
        {
            'name': "",
            'type': "Stopping",
            'args': {
                'body': "",
                'max_linvel': STOPPING_LINEAR_VELOCITY,
                'max_angvel': STOPPING_ANGULAR_VELOCITY,
            }
        },
    ]
}
