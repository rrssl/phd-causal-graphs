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
GOBLET_MASS = .006  # [g]
GOBLET_R1 = 0.036  # [m]
GOBLET_R2 = 0.025  # [m]
LEFT_PULLEY_ROPE_LENGTH = .95  # [m]
LONG_TRACK_LWHT = (0.3, 0.025, 0.008, .002)  # [m]
LONG_TRACK_MASS = .01  # [kg]
NAIL_LEVER_LWH = (0.11, 0.005, 0.002)  # [m]
NAIL_LEVER_MASS = 0.005  # [kg]
PIVOT_LENGTH = .06  # [m]
PIVOT_RADIUS = .004  # [m]
PLANK_LWH = (0.1175, 0.023, 0.008)  # [m]
PLANK_MASS = 0.01  # [kg]
PLANK_RESTITUTION = 0.8
QUAD_PLANK_LWH = (0.031, 0.023, 0.1175)  # [m]
RIGHT_PULLEY_PIVOT_HEIGHT = .003  # [m]
RIGHT_PULLEY_PIVOT_RADIUS = .006  # [m]
RIGHT_PULLEY_ROPE_LENGTH = 1.  # [m]
RIGHT_WEIGHT_HEIGHT = 0.0315  # [m]
RIGHT_WEIGHT_MASS = 0.2  # [kg]
RIGHT_WEIGHT_RADIUS = 0.0315 / 2  # [m]
SHORT_TRACK_LWHT = (0.15, 0.025, 0.008, .002)  # [m]
SMOL_TRACK_LWHT = (0.1, 0.025, 0.008, .002)  # [m]
SMOL_WEIGHT_HEIGHT = 0.014  # [m]
SMOL_WEIGHT_RADIUS = 0.0075  # [m]
SMOL_WEIGHT_MASS = 0.02  # [kg]
STARTING_TRACK_FRICTION = 1
STARTING_TRACK_LWHT = (0.3, 0.025, 0.005, .002)  # [m]
TEAPOT_FRICTION = .1
TEAPOT_LID_ANGULAR_DAMPING = 1
TEAPOT_LID_HEIGHT = .005  # [m]
TEAPOT_LID_MASS = 0.002  # [kg] TODO
TEAPOT_LID_RADIUS = GOBLET_R1
TINY_TRACK_LWH = (0.1, 0.025, 0.005)  # [m]
TINY_TRACK_MASS = .01  # [kg] TODO
GATE_PULLEY_ROPE_LENGTH = .50  # [m]
GATE_PULLEY_ROPE_EXTENTS = (.002, .5, .01)   # [m]
PULLEY_EXTENTS = (.005, .02)  # [m]
GATE_PUSHER_GUIDE_LWHT = (0.099, 0.016, 0.012, 0.003)  # [m]

PIVOTING_ANGULAR_VELOCITY = 1
ANGVEL = 10
STOPPING_LINEAR_VELOCITY = 1e-2
STOPPING_ANGULAR_VELOCITY = 1

DATA = {
    'scene': [
        {
            'name': "start_track_origin",
            'type': "Empty",
            'args': {},
            'xform': {
                'value': [-.33, 0, .105, 0, 0, 11],
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
                'value': [-.13, 0, BALL_RADIUS+.002, 0, 0, 0]
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
                'value': [-.10, 0, BALL_RADIUS+.002, 0, 0, 0]
            }
        },
        {
            'name': "cage",
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
                'value': [-.32, 0, .21, 0, 0, 180],
            }
        },
        {
            'name': "cage_support",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
            },
            'xform': {
                'value': [-.32, 0, .094, 90, 0, 0]
            }
        },
        {
            'name': "gate",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
                'b_mass': PLANK_MASS,
            },
            'xform': {
                'value': [0, 0, .15, 0, 0, 90],
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
                'pulleys': [[-.32, 0, .29], [0, 0, .29]],
                'pulley_extents': PULLEY_EXTENTS
            },
            'components': ["cage", "gate"],
        },
        {
            'name': "gate_guide_left",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
            },
            'xform': {
                'value': [-.011, 0, .15, 0, 0, 90],
            }
        },
        {
            'name': "gate_guide_back",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
            },
            'xform': {
                'value': [0, .02, .15, 90, 0, 90],
            }
        },
        {
            'name': "gate_guide_right",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
            },
            'xform': {
                'value': [.01, 0, .18, 0, 0, 90],
            }
        },
        {
            'name': "gate_pusher_guide",
            'type': "Track",
            'args': {
                'extents': GATE_PUSHER_GUIDE_LWHT,
            },
            'xform': {
                'value': [PLANK_LWH[0]/2-.007, 0, .08, 0, 0, 0],
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
                'value': [.011, 0, .009, 0, 90, 0],
            }
        },
        {
            'name': "hit_lever",
            'type': "Lever",
            'args': {
                'extents': PLANK_LWH,
                'pivot_pos': [0, 0, .005],
                'pivot_hpr': [0, 90, 0],
                'pivot_extents': [PIVOT_RADIUS, PIVOT_LENGTH],
                'b_mass': PLANK_MASS
            },
            'parent': "gate_pusher_guide",
            'xform': {
                'value': [.075, 0, -.04, 0, 0, 90],
            }
        },
        {
            'name': "middle_track",
            'type': "Track",
            'args': {
                'extents': SMOL_TRACK_LWHT,
            },
            'xform': {
                'value': [.05, 0, 0, 0, 0, 0],
            }
        },
        {
            'name': "right_track_top",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [.16, 0, -.05, 0, 0, 3],
            }
        },
        {
            'name': "right_track_middle",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [.26, 0, -.12, 0, 0, -5],
            }
        },
        {
            'name': "balance_track",
            'type': "Track",
            'args': {
                'extents': STARTING_TRACK_LWHT,
                'b_mass': LONG_TRACK_MASS
            },
            'xform': {
                'value': [.16, 0, -.17, 0, 0, 0],
            }
        },
        {
            'name': "balance_track_left_weight",
            'type': "Cylinder",
            'args': {
                'extents': [SMOL_WEIGHT_RADIUS, SMOL_WEIGHT_HEIGHT],
                'b_mass': SMOL_WEIGHT_MASS
            },
            'parent': "balance_track",
            'xform': {
                'value': [-STARTING_TRACK_LWHT[0]/2+.005,
                          0,
                          -STARTING_TRACK_LWHT[2]/2+STARTING_TRACK_LWHT[3]+(
                              SMOL_WEIGHT_HEIGHT/2),
                          0, 0, 0],
            }
        },
        {
            'name': "balance_track_right_weight",
            'type': "Cylinder",
            'args': {
                'extents': [SMOL_WEIGHT_RADIUS, SMOL_WEIGHT_HEIGHT],
                'b_mass': SMOL_WEIGHT_MASS
            },
            'parent': "balance_track",
            'xform': {
                'value': [STARTING_TRACK_LWHT[0]/2-.01,
                          0,
                          -STARTING_TRACK_LWHT[2]/2-SMOL_WEIGHT_HEIGHT/2,
                          0, 0, 0],
            }
        },
        {
            'name': "fastener1",
            'type': "Fastener",
            'args': {
                'comp1_xform': [
                    STARTING_TRACK_LWHT[0]/2-.01, 0, -STARTING_TRACK_LWHT[2]/2,
                    0, 0, 0
                ],
                'comp2_xform': [0, 0, SMOL_WEIGHT_HEIGHT/2, 0, 0, 0]
            },
            'components': ["balance_track", "balance_track_right_weight"]
        },
        {
            'name': "balance_track_pivot",
            'type': "Pivot",
            'args': {
                'pivot_pos': [0,
                              0,
                              -STARTING_TRACK_LWHT[2]/2-.003],
                'pivot_hpr': [0, 90, 0],
                'pivot_extents': [PIVOT_RADIUS, PIVOT_LENGTH],
            },
            'components': ["balance_track"],
        },
        {
            'name': "balance_track_blocker",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH
            },
            'xform': {
                'value': [.20, 0, -.19, 90, 0, 0],
            }
        },
        {
            'name': "bridge_blocker",
            'type': "Box",
            'args': {
                'extents': FLAT_SUPPORT_LWH,
                'b_mass': .001
            },
            'parent': "balance_track",
            'xform': {
                'value': [-STARTING_TRACK_LWHT[0]/3,
                          0,
                          -STARTING_TRACK_LWHT[2]/2-FLAT_SUPPORT_LWH[0]/2,
                          0, 0, -90],
            }
        },
        {
            'name': "fastener2",
            'type': "Fastener",
            'args': {
                'comp1_xform': [
                    -STARTING_TRACK_LWHT[0]/3, 0, -STARTING_TRACK_LWHT[2]/2,
                    0, 0, 0
                ],
                'comp2_xform': [FLAT_SUPPORT_LWH[0]/2, 0, 0, 0, 0, 90]
            },
            'components': ["balance_track", "bridge_blocker"]
        },
        {
            'name': "right_track_bottom",
            'type': "Track",
            'args': {
                'extents': LONG_TRACK_LWHT,
            },
            'xform': {
                'value': [.35, 0, -.26, 0, 0, -10],
            }
        },
        {
            'name': "left_track_top_origin",
            'type': "Empty",
            'args': {},
            'parent': "middle_track",
            'xform': {
                'value': [-.03, 0, -.01, 0, 0, -10],
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
            'name': "left_track_middle_top_origin",
            'type': "Empty",
            'args': {},
            'parent': "cap_counterweight_support",
            'xform': {
                'value': [FLAT_SUPPORT_LWH[0]/2, 0, FLAT_SUPPORT_LWH[2]/2,
                          0, 0, 10],
            }
        },
        {
            'name': "left_track_middle_top",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'parent': "left_track_middle_top_origin",
            'xform': {
                'value': [SHORT_TRACK_LWHT[0]/2, 0, SHORT_TRACK_LWHT[2]/2,
                          0, 0, 0],
            }
        },
        {
            'name': "left_track_middle_bottom",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
            },
            'xform': {
                'value': [-.15, 0, -.20, 0, 0, -10],
            }
        },
        {
            'name': "left_track_bottom",
            'type': "Track",
            'args': {
                'extents': LONG_TRACK_LWHT,
            },
            'xform': {
                'value': [-.20, 0, -.25, 0, 0, 10],
            }
        },
        {
            'name': "cap_counterweight_support",
            'type': "Box",
            'args': {
                'extents': FLAT_SUPPORT_LWH
            },
            'xform': {
                'value': [-.35, 0, -.15, 0, 0, 0]
            }
        },
        {
            'name': "cap_counterweight",
            'type': "Box",
            'args': {
                'extents': QUAD_PLANK_LWH,
                'b_mass': 4*PLANK_MASS,
                'b_restitution': PLANK_RESTITUTION
            },
            'parent': "cap_counterweight_support",
            'xform': {
                'value': [-FLAT_SUPPORT_LWH[0]/2+.001,
                          0,
                          QUAD_PLANK_LWH[2]/2+FLAT_SUPPORT_LWH[2]/2,
                          0, 0, 0]
            }
        },
        {
            'name': "cap_counterweight_support_side",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH
            },
            'parent': "cap_counterweight_support",
            'xform': {
                'value': [-FLAT_SUPPORT_LWH[0]/2+PLANK_LWH[2]/2,
                          0,
                          -PLANK_LWH[0]/2-FLAT_SUPPORT_LWH[2]/2,
                          0, 0, 90]
            }
        },
        {
            'name': "cap_counterweight_support_bottom",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH
            },
            'parent': "cap_counterweight_support_side",
            'xform': {
                'value': [PLANK_LWH[0]/2-PLANK_LWH[2]/2,
                          0,
                          -PLANK_LWH[0]/2-PLANK_LWH[2]/2,
                          0, 0, 90]
            }
        },
        {
            'name': "bridge",
            'type': "Track",
            'args': {
                'extents': SHORT_TRACK_LWHT,
                'b_mass': PLANK_MASS
            },
            'xform': {
                'value': [.09, 0, -.25, 0, 0, 80],
            }
        },
        {
            'name': "bridge_pivot",
            'type': "Pivot",
            'args': {
                'pivot_pos': [SHORT_TRACK_LWHT[0]/2-.002,
                              0,
                              -SHORT_TRACK_LWHT[2]/2-.004],
                'pivot_hpr': [0, 90, 0],
                'pivot_extents': [PIVOT_RADIUS, PIVOT_LENGTH],
            },
            'components': ["bridge"],
        },
        {
            'name': "bridge_support",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH
            },
            'xform': {
                'value': [-.05, 0, -.29, 90, 0, 0],
            }
        },
        {
            'name': "pit",
            'type': "Goblet",
            'args': {
                'extents': [
                    GOBLET_HEIGHT,
                    GOBLET_R1,
                    GOBLET_R2
                ],
            },
            'xform': {
                'value': [-.02, 0, -.45, 0, 0, 0]
            }
        },
        {
            'name': "teapot",
            'type': "Goblet",
            'args': {
                'extents': [
                    GOBLET_HEIGHT,
                    GOBLET_R1,
                    GOBLET_R2
                ],
            },
            'xform': {
                'value': [.15, 0, -.45, 0, 0, 0]
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
