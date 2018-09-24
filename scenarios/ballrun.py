from math import atan, degrees

BALL_RADIUS = 0.015 / 2  # [m]
BALL_MASS = 0.01 / 6  # [kg]
TOP_TRACK_LWHT = (0.3, 0.025, 0.005, 0.001)  # [m]
BOTTOM_TRACK_LWHT = TOP_TRACK_LWHT  # [m]
HIGH_PLANK_LWH = (0.235, 0.023, 0.008)  # [m]
HIGH_PLANK_MASS = 0.021  # [kg]
LOW_PLANK_LWH = (0.1175, 0.023, 0.016)  # [m]
LOW_PLANK_MASS = 0.02  # [kg]
BASE_PLANK_LWH = (0.35, 0.025, 0.005)  # [m]
BASE_PLANK_MASS = 0.021  # [kg]
# ROUND_SUPPORT_RADIUS = 0.014  # [m]
# ROUND_SUPPORT_HEIGHT = 0.04   # [m]
FLAT_SUPPORT_LWH = (.02, .025, .005)  # [m]
GOBLET_HEIGHT = 0.119  # [m]
GOBLET_R1 = 0.0455  # [m]
GOBLET_R2 = 0.031  # [m]

PIVOTING_ANGULAR_VELOCITY = 1
ROLLING_ANGLE = 90
HIGH_PLANK_TOPPLING_ANGLE = degrees(atan(HIGH_PLANK_LWH[2]/HIGH_PLANK_LWH[0]))
STOPPING_LINEAR_VELOCITY = 1e-2
STOPPING_ANGULAR_VELOCITY = 1

DATA = {
    'scene': [
        {
            'name': "board",
            'type': "Plane",
            'args': {
                'normal': [0, 1, 0],
                'distance': 0
            }
        },
        {
            'name': "ball",
            'type': "Ball",
            'args': {
                'radius': BALL_RADIUS,
                'b_mass': BALL_MASS,
                'b_restitution': .5
            },
            'parent': "top_track",
            'xform': {
                'value': [-TOP_TRACK_LWHT[0]/2, 0, BALL_RADIUS,
                          0, 0, 0]
            }
        },
        {
            'name': "top_track",
            'type': "Track",
            'args': {
                'extents': TOP_TRACK_LWHT,
            },
            'xform': {
                'value': [-.1, TOP_TRACK_LWHT[1]/2+.01, .3, 0, 0, 20],
                'range': [
                    [-BASE_PLANK_LWH[0]/2, BASE_PLANK_LWH[0]/4],
                    None,
                    [HIGH_PLANK_LWH[0]/2,
                     HIGH_PLANK_LWH[0]+TOP_TRACK_LWHT[0]/2],
                    None,
                    None,
                    [5, 45]
                ]
            }
        },
        {
            'name': "bottom_track",
            'type': "Track",
            'args': {
                'extents': BOTTOM_TRACK_LWHT,
            },
            'xform': {
                'value': [0, TOP_TRACK_LWHT[1]/2+.01, .1, 0, 0, -10],
                'range': [
                    None,
                    None,
                    [LOW_PLANK_LWH[0]/2, HIGH_PLANK_LWH[0]],
                    None,
                    None,
                    [-45, -5]
                ]
            }
        },
        {
            'name': "high_plank",
            'type': "Box",
            'args': {
                'extents': HIGH_PLANK_LWH,
                'b_mass': HIGH_PLANK_MASS,
                'b_restitution': .5
            },
            'parent': "base_plank",
            'xform': {
                'value': [
                    BOTTOM_TRACK_LWHT[0]/2 + HIGH_PLANK_LWH[2]/2,
                    0,
                    HIGH_PLANK_LWH[0]/2 + BASE_PLANK_LWH[2]/2,
                    0, 0, 90
                ],
            }
        },
        {
            'name': "low_plank",
            'type': "Box",
            'args': {
                'extents': LOW_PLANK_LWH,
                'b_mass': LOW_PLANK_MASS
            },
            'parent': "base_plank",
            'xform': {
                'value': [
                    -BOTTOM_TRACK_LWHT[0]/2 - LOW_PLANK_LWH[2]/2,
                    0,
                    LOW_PLANK_LWH[0]/2 + BASE_PLANK_LWH[2]/2,
                    0, 0, 90
                ],
            }
        },
        {
            'name': "base_plank",
            'type': "Lever",
            'args': {
                'extents': BASE_PLANK_LWH,
                'pivot_pos': [0, 0, 0],
                'pivot_hpr': [0, 90, 0],
                'pivot_extents': [BASE_PLANK_LWH[2]*.4,
                                  BASE_PLANK_LWH[1]*1.5],
                'b_mass': BASE_PLANK_MASS
            },
            'xform': {
                'value': [
                    0,
                    BASE_PLANK_LWH[1]/2 + .01,
                    0,
                    # ROUND_SUPPORT_RADIUS + BASE_PLANK_LWH[2]/2,
                    0, 0, 0
                ],
            }
        },
        {
            'name': "flat_support",
            'type': "Box",
            'args': {
                'extents': FLAT_SUPPORT_LWH,
            },
            'xform': {
                'value': [
                    BASE_PLANK_LWH[0]/2 - FLAT_SUPPORT_LWH[0]/2,
                    FLAT_SUPPORT_LWH[0]/2 + .01,
                    -FLAT_SUPPORT_LWH[2],
                    0, 0, 0
                ],
            }
        },
        # {
        #     'name': "round_support",
        #     'type': "Cylinder",
        #     'args': {
        #         'extents': [
        #             ROUND_SUPPORT_RADIUS,
        #             ROUND_SUPPORT_HEIGHT
        #         ]
        #     },
        #     'xform': {
        #         'value': [0, ROUND_SUPPORT_HEIGHT/2, 0, 0, 90, 0],
        #     }
        # },
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
                'value': [-.35, GOBLET_R1, -.05, 0, 0, 45],
                'range': [
                    [-BASE_PLANK_LWH[0],
                     -BASE_PLANK_LWH[0]/2-GOBLET_R1/2],
                    None,
                    [-LOW_PLANK_LWH[0]/2, LOW_PLANK_LWH[0]/2],
                    None,
                    None,
                    [0, 60]
                ]
            }
        },
    ],
    'causal_graph': [
        {
            'name': "ball_rolls_on_top_track",
            'type': "RollingOn",
            'args': {
                'rolling': "ball",
                'support': "top_track",
                'min_rollang': ROLLING_ANGLE
            },
            'children': [
                "ball_hits_high_plank",
            ]
        },
        {
            'name': "ball_hits_high_plank",
            'type': "Contact",
            'args': {
                'first': "ball",
                'second': "high_plank"
            },
            'children': [
                "high_plank_topples",
                "ball_rolls_on_bottom_track",
            ]
        },
        {
            'name': "high_plank_topples",
            'type': "Toppling",
            'args': {
                'body': "high_plank",
                'angle': HIGH_PLANK_TOPPLING_ANGLE,
            },
            'children': [
                "base_plank_moves",
            ]
        },
        {
            'name': "ball_rolls_on_bottom_track",
            'type': "RollingOn",
            'args': {
                'rolling': "ball",
                'support': "bottom_track",
                'min_rollang': ROLLING_ANGLE
            },
            'children': [
                "ball_enters_goblet",
            ]
        },
        {
            'name': "base_plank_moves",
            'type': "Pivoting",
            'args': {
                'body': "base_plank",
                'min_angvel': PIVOTING_ANGULAR_VELOCITY,
            },
            'children': [
                "low_plank_falls",
            ]
        },
        {
            'name': "low_plank_falls",
            'type': "NoContact",
            'args': {
                'first': "low_plank",
                'second': "base_plank",
            },
            'children': [
                "ball_enters_goblet",
            ]
        },
        {
            'name': "ball_enters_goblet",
            'type': "Inclusion",
            'args': {
                'inside': "ball",
                'outside': "goblet",
            },
            'children': [
                "ball_stops",
            ]
        },
        {
            'name': "ball_stops",
            'type': "Stopping",
            'args': {
                'body': "ball",
                'max_linvel': STOPPING_LINEAR_VELOCITY,
                'max_angvel': STOPPING_ANGULAR_VELOCITY,
            }
        },
    ]
}
