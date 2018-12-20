F = 1
BALL_FRICTION = .1
# BALL_MASS = 0.013  # [kg]
BALL_MASS = 0.0056  # [kg]
# BALL_RADIUS = 0.0105  # [m]
BALL_RADIUS = 0.008*F  # [m]
BALL_RESTITUTION = 0.8
FLAT_SUPPORT_LWH = (.02*F, .025*F, .005*F)  # [m]
GOBLET_ANGULAR_DAMPING = 1
GOBLET_FRICTION = 1
GOBLET_HEIGHT = 0.11*F  # [m]
GOBLET_MASS = .005  # [g]
GOBLET_R1 = 0.036*F  # [m]
GOBLET_R2 = 0.025*F  # [m]
LEFT_PULLEY_ROPE_LENGTH = .95*F  # [m]
LONG_TRACK_LWHT = (0.3*F, 0.025*F, 0.008*F, .002*F)  # [m]
NAIL_LEVER_LWH = (0.11*F, 0.005*F, 0.002*F)  # [m]
NAIL_LEVER_MASS = 0.005  # [kg]
PLANK_LWH = (0.1175*F, 0.023*F, 0.008*F)  # [m]
PLANK_MASS = 0.01  # [kg]
PLANK_RESTITUTION = 0.8
QUAD_PLANK_LWH = (0.1175*F, 0.023*F, 0.031*F)  # [m]
RIGHT_PULLEY_PIVOT_HEIGHT = .003*F  # [m]
RIGHT_PULLEY_PIVOT_RADIUS = .006*F  # [m]
RIGHT_PULLEY_ROPE_LENGTH = 1.*F  # [m]
RIGHT_WEIGHT_HEIGHT = 0.0315*F  # [m]
RIGHT_WEIGHT_MASS = 0.2  # [kg]
RIGHT_WEIGHT_RADIUS = 0.0315*F / 2  # [m]
SHORT_TRACK_LWHT = (0.15*F, 0.025*F, 0.008*F, .002*F)  # [m]
STARTING_TRACK_FRICTION = 1
STARTING_TRACK_LWHT = (0.3*F, 0.025*F, 0.005*F, .002*F)  # [m]
TEAPOT_FRICTION = .1
TEAPOT_LID_ANGULAR_DAMPING = 1
TEAPOT_LID_HEIGHT = .005*F  # [m]
TEAPOT_LID_MASS = 0.002  # [kg] TODO
TEAPOT_LID_RADIUS = GOBLET_R1
TINY_TRACK_LWH = (0.1*F, 0.025*F, 0.005*F)  # [m]
TINY_TRACK_MASS = .01  # [kg] TODO
GATE_PULLEY_ROPE_LENGTH = .50*F  # [m]
GATE_PULLEY_ROPE_EXTENTS = (.002*F, .5*F, .01*F)   # [m]
PULLEY_EXTENTS = (.005*F, .02*F)  # [m]
GATE_PUSHER_GUIDE_LWHT = (0.099*F, 0.016*F, 0.012*F, 0.003*F)  # [m]

PIVOTING_ANGULAR_VELOCITY = 1
ANGVEL = 10
STOPPING_LINEAR_VELOCITY = 1e-2
STOPPING_ANGULAR_VELOCITY = 1

# Track("middle_track", SHORT_TRACK_LWHT,)
# Track("left_track1", LONG_TRACK_LWHT,)
# Track("left_track2", SHORT_TRACK_LWHT,)
# Track("left_track3", SHORT_TRACK_LWHT,)
# Box("left_track3_blocker", FLAT_SUPPORT_LWH,)
# Track("left_track4", LONG_TRACK_LWHT,)
# Box("left_track4_blocker", FLAT_SUPPORT_LWH,)
# Track("right_track1", LONG_TRACK_LWHT,)
# Track("right_track2", SHORT_TRACK_LWHT,)
# Box("right_track2_blocker1", FLAT_SUPPORT_LWH,)
# Box("right_track2_blocker2", FLAT_SUPPORT_LWH,)
# Track("right_track3", SHORT_TRACK_LWHT,)
# Box("right_track3_blocker", FLAT_SUPPORT_LWH,)
# Track("right_track4", SHORT_TRACK_LWHT,)
# Box("right_track4_blocker", FLAT_SUPPORT_LWH,)
# Box("top_weight_support", TINY_TRACK_LWH,)
# Box("top_weight_guide", FLAT_SUPPORT_LWH,)
# Box("left_weight_support", FLAT_SUPPORT_LWH,)
# Box("right_weight_support", FLAT_SUPPORT_LWH,)
# Goblet("top_goblet", (GOBLET_HEIGHT, GOBLET_R1, GOBLET_R2), mass=GOBLET_MASS,
#        friction=GOBLET_FRICTION, angular_damping=GOBLET_ANGULAR_DAMPING)
# Goblet("bottom_goblet", (GOBLET_HEIGHT, GOBLET_R1, GOBLET_R2),)
# Box("sliding_plank", PLANK_LWH, mass=PLANK_MASS)
# Box("gate", PLANK_LWH, mass=PLANK_MASS)
# Box("left_weight", QUAD_PLANK_LWH, mass=4*PLANK_MASS,)
# Cylinder("right_weight", (RIGHT_WEIGHT_RADIUS, RIGHT_WEIGHT_HEIGHT),
#          mass=RIGHT_WEIGHT_MASS)
# Box("bridge", TINY_TRACK_LWH, mass=TINY_TRACK_MASS)
# Goblet("teapot_base", (GOBLET_HEIGHT, GOBLET_R1, GOBLET_R2),
#        friction=TEAPOT_FRICTION)
# Cylinder("teapot_lid", (TEAPOT_LID_RADIUS, TEAPOT_LID_HEIGHT),
#          mass=TEAPOT_LID_MASS, angular_damping=TEAPOT_LID_ANGULAR_DAMPING)
# Box("nail", NAIL_LEVER_LWH, mass=NAIL_LEVER_MASS)
# Pivot("pivot", nail, (-NAIL_LEVER_LWH[0]*.2, 0, 0),  # magical (0, 90, 0),)
# RopePulley("top_pulley", top_goblet, gate, (0), (-PLANK_LWH[0]/2, 0, 0),
#            TOP_PULLEY_ROPE_LENGTH,)
# RopePulley("left_pulley", left_weight, teapot_lid, (-PLANK_LWH[0]/2, 0, 0),
#            (0, 0, TEAPOT_LID_HEIGHT)/2, LEFT_PULLEY_ROPE_LENGTH,)

DATA = {
    'scene': [
        # {
        #     'name': "",
        #     'type': "",
        #     'args': {
        #         '': 0,
        #     }
        # },
        # {
        #     'name': "",
        #     'type': "",
        #     'args': {
        #         '': 0,
        #     },
        #     'parent': "",
        #     'xform': {
        #         'value': [],
        #         'range': [
        #             [],
        #             None,
        #             [],
        #             None,
        #             None,
        #             []
        #         ]
        #     }
        # },
        {
            'name': "start_track_origin",
            'type': "Empty",
            'args': {},
            'xform': {
                'value': [-.33*F, 0, .10*F, 0, 0, 10],
                'range': [
                    None,
                    None,
                    None,
                    None,
                    None,
                    [5, 15]
                ]
            }
        },
        {
            'name': "start_track",
            'type': "Track",
            'args': {
                'extents': STARTING_TRACK_LWHT,
            },
            'parent': "start_track_origin",
            'xform': {
                'value': [STARTING_TRACK_LWHT[0]/2, 0, 0, 0, 0, 0],
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
            'parent': "start_track",
            'xform': {
                'value': [-.13*F, 0, BALL_RADIUS+.002*F, 0, 0, 0]
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
            'parent': "start_track",
            'xform': {
                'value': [-.10*F, 0, BALL_RADIUS+.002*F, 0, 0, 0]
            }
        },
        {
            'name': "gate_goblet",
            'type': "Goblet",
            'args': {
                'extents': [
                    GOBLET_HEIGHT,
                    GOBLET_R1,
                    GOBLET_R2
                ],
                'b_mass': GOBLET_MASS,
                'b_friction': GOBLET_FRICTION,
                'b_angular_damping': GOBLET_ANGULAR_DAMPING,
            },
            'xform': {
                'value': [-.32*F, 0, .21*F, 0, 0, 180],
            }
        },
        {
            'name': "gate",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
                'b_mass': PLANK_MASS,
                'b_restitution': PLANK_RESTITUTION
            },
            'xform': {
                'value': [0, 0, .15*F, 0, 0, 90],
            }
        },
        {
            'name': "gate_pulley",
            'type': "RopePulley",
            'args': {
                'comp1_pos': [0, 0, 0],
                'comp2_pos': [-PLANK_LWH[0]/2, 0, 0],
                # 'rope_extents': GATE_PULLEY_ROPE_EXTENTS,
                'rope_length': GATE_PULLEY_ROPE_LENGTH,
                'pulleys': [[-.32*F, 0, .29*F], [0, 0, .29*F]],
                'pulley_extents': PULLEY_EXTENTS
            },
            'components': ["gate_goblet", "gate"],
        },
        {
            'name': "gate_guide_left",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
            },
            'xform': {
                'value': [-.01*F, 0, .15*F, 0, 0, 90],
            }
        },
        {
            'name': "gate_guide_back",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
            },
            'xform': {
                'value': [0*F, .02*F, .15*F, 90, 0, 90],
            }
        },
        {
            'name': "gate_guide_right",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
            },
            'xform': {
                'value': [.01*F, 0, .18*F, 0, 0, 90],
            }
        },
        {
            'name': "gate_pusher_guide",
            'type': "Track",
            'args': {
                'extents': GATE_PUSHER_GUIDE_LWHT,
            },
            'xform': {
                'value': [PLANK_LWH[0]/2-.007*F, 0, .08*F, 0, 0, 0],
            }
        },
        {
            'name': "gate_pusher",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
                'b_mass': PLANK_MASS
            },
            'parent': "gate_pusher_guide",
            'xform': {
                'value': [.011*F, 0, .009*F, 0, 90, 0],
            }
        },
        {
            'name': "hit_lever",
            'type': "Lever",
            'args': {
                'extents': PLANK_LWH,
                'pivot_pos': [0, 0, .005*F],
                'pivot_hpr': [0, 90, 0],
                'pivot_extents': [.004*F, .03*F],
                'b_mass': PLANK_MASS
            },
            'parent': "gate_pusher_guide",
            'xform': {
                'value': [.08*F, 0, -.04*F, 0, 0, 90],
            }
        },
        {
            'name': "middle_track",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [.02*F, 0, 0, 0, 0, 0],
            }
        },
        {
            'name': "right_track_top",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [.16*F, 0, -.05*F, 0, 0, 5],
            }
        },
        {
            'name': "right_track_middle",
            'type': "Track",
            'args': {
                'extents': LONG_TRACK_LWHT,
            },
            'xform': {
                'value': [.17*F, 0, -.11*F, 0, 0, -12],
            }
        },
        {
            'name': "right_track_bottom",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [.16*F, 0, -.20*F, 0, 0, -10],
            }
        },
        {
            'name': "left_track_top_origin",
            'type': "Empty",
            'args': {},
            'parent': "middle_track",
            'xform': {
                'value': [-SHORT_TRACK_LWHT[0]/2, 0, 0*F, 0, 0, -10],
            }
        },
        {
            'name': "left_track_top",
            'type': "Track",
            'args': {
                'extents': LONG_TRACK_LWHT,
            },
            'parent': "left_track_top_origin",
            'xform': {
                'value': [-LONG_TRACK_LWHT[0]/2, 0, 0, 0, 0, 0],
            }
        },
        {
            'name': "left_track_middle_top",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [-.3*F, 0, -.1*F, 0, 0, 10],
            }
        },
        {
            'name': "left_track_middle_bottom",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [-.25*F, 0, -.15*F, 0, 0, -10],
            }
        },
        {
            'name': "left_track_bottom",
            'type': "Track",
            'args': {
                'extents': LONG_TRACK_LWHT,
            },
            'xform': {
                'value': [-.25*F, 0, -.20*F, 0, 0, 10],
            }
        },
    ],
    'causal_graph': [
        # {
        #     'name': "",
        #     'type': "RollingOn",
        #     'args': {
        #         'rolling': "",
        #         'support': "",
        #         'min_angvel': 0
        #     },
        #     'children': [
        #         "",
        #     ]
        # },
        # {
        #     'name': "",
        #     'type': "Contact",
        #     'args': {
        #         'first': "",
        #         'second': ""
        #     },
        #     'children': [
        #         "",
        #         "",
        #     ]
        # },
        # {
        #     'name': "",
        #     'type': "",
        #     'args': {
        #         'body': "",
        #         'angle': 0,
        #     },
        #     'children': [
        #         "",
        #     ]
        # },
        # {
        #     'name': "",
        #     'type': "Pivoting",
        #     'args': {
        #         'body': "",
        #         'min_angvel': 0,
        #     },
        #     'children': [
        #         "",
        #     ]
        # },
        # {
        #     'name': "",
        #     'type': "NoContact",
        #     'args': {
        #         'first': "",
        #         'second': "",
        #     },
        #     'children': [
        #         "",
        #     ]
        # },
        # {
        #     'name': "",
        #     'type': "Inclusion",
        #     'args': {
        #         'inside': "",
        #         'outside': "goblet",
        #     },
        #     'children': [
        #         "",
        #     ]
        # },
        # {
        #     'name': "",
        #     'type': "Stopping",
        #     'args': {
        #         'body': "",
        #         'max_linvel': STOPPING_LINEAR_VELOCITY,
        #         'max_angvel': STOPPING_ANGULAR_VELOCITY,
        #     }
        # },
    ]
}
