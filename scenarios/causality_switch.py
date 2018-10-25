import os
import sys

sys.path.insert(0, os.path.abspath(".."))
from core.dominoes import (create_branch, create_line,  # noqa: E402
                           create_wave, create_x_switch)


SUPPORT_LWH = [.175, .33, .093]
DOMINO_LWH = [.007, .02, .044]
DOMINO_MASS = .00305
PLANK_LWH = [.024, .008, .118]
PLANK_MASS = .01
TRACK_LWHT = [.30, .024, .005, .001]
BALL_RADIUS = .01
BALL_MASS = .01
BRANCH_LENGTH = .20
BRANCH_WIDTH = .12
STRAIGHT_LENGTH = .13
WAVE_LENGTH = .4
WAVE_WIDTH = .08
SWITCH_WIDTH = BRANCH_WIDTH

LEFT_FASTER = 1
if LEFT_FASTER:
    FASTER_SIDE = 'left'
    SLOWER_SIDE = 'right'

DATA = {
    'scene': [
        {
            'name': "ground",
            'type': "Plane",
            'args': {
                'normal': [0, 0, 1],
                'distance': 0
            },
        },
        {
            'name': "support",
            'type': "Box",
            'args': {
                'extents': SUPPORT_LWH
            },
            'xform': {
                'value': [0, -SUPPORT_LWH[1]/2, SUPPORT_LWH[2]/2, 0, 0, 0]
            }
        },
        {
            'name': "branch_run",
            'type': "DominoRun",
            'args': {
                'extents': DOMINO_LWH,
                'coords': create_branch([0, 0], 0,
                                        BRANCH_LENGTH, BRANCH_WIDTH, 5),
                'tilt_angle': 10,
                'b_mass': DOMINO_MASS
            },
            'parent': "support",
            'xform': {
                'value': [0, -.08, SUPPORT_LWH[2]/2, 90, 0, 0]
            }
        },
        {
            'name': "smol_run",
            'type': "DominoRun",
            'args': {
                'extents': DOMINO_LWH,
                'coords': create_line([0, 0], 0, .02, 2),
                'b_mass': DOMINO_MASS
            },
            'parent': "support",
            'xform': {
                'value': [BRANCH_WIDTH/2,
                          SUPPORT_LWH[1]/2-.025,
                          SUPPORT_LWH[2]/2,
                          90, 0, 0]
            }
        },
        {
            'name': "ball",
            'type': "Ball",
            'args': {
                'radius': BALL_RADIUS,
                'b_mass': BALL_MASS
            },
            'parent': "support",
            'xform': {
                'value': [-BRANCH_WIDTH/2,
                          SUPPORT_LWH[1]/2-BALL_RADIUS/2,
                          BALL_RADIUS+SUPPORT_LWH[2]/2,
                          0, 0, 0]
            }
        },
        {
            'name': "ball_guide1",
            'type': "Box",
            'args': {
                'extents': [DOMINO_LWH[1], DOMINO_LWH[2], DOMINO_LWH[0]],
            },
            'parent': "support",
            'xform': {
                'value': [-BRANCH_WIDTH/2-DOMINO_LWH[1]-.003,
                          SUPPORT_LWH[1]/2-DOMINO_LWH[2]/2,
                          DOMINO_LWH[0]/2+SUPPORT_LWH[2]/2,
                          0, 0, 0]
            }
        },
        {
            'name': "ball_guide2",
            'type': "Box",
            'args': {
                'extents': [DOMINO_LWH[1], DOMINO_LWH[2], DOMINO_LWH[0]],
            },
            'parent': "support",
            'xform': {
                'value': [-BRANCH_WIDTH/2+DOMINO_LWH[1]+.003,
                          SUPPORT_LWH[1]/2-DOMINO_LWH[2]/2,
                          DOMINO_LWH[0]/2+SUPPORT_LWH[2]/2,
                          0, 0, 0]
            }
        },
        {
            'name': "track",
            'type': "Track",
            'args': {
                'extents': TRACK_LWHT,
            },
            'xform': {
                'value': [-BRANCH_WIDTH/2,
                          TRACK_LWHT[0]/2-.004,
                          SUPPORT_LWH[2]/2+.008,
                          90, 0, 15]
            }
        },
        {
            'name': "plank",
            'type': "Box",
            'args': {
                'extents': PLANK_LWH,
                'b_mass': PLANK_MASS
            },
            'xform': {
                'value': [BRANCH_WIDTH/2, .01, PLANK_LWH[2]/2, 0, 0, 0]
            }
        },
        {
            'name': "straight_run",
            'type': "DominoRun",
            'args': {
                'extents': DOMINO_LWH,
                'coords': create_line([0, 0], 0, STRAIGHT_LENGTH, 7),
                'b_mass': DOMINO_MASS
            },
            'xform': {
                'value': [-BRANCH_WIDTH/2, .31, 0, 90, 0, 0]
            }
        },
        {
            'name': "wavey_run",
            'type': "DominoRun",
            'args': {
                'extents': DOMINO_LWH,
                'coords': create_wave([0, 0], 0, WAVE_LENGTH, WAVE_WIDTH, 20),
                'b_mass': DOMINO_MASS
            },
            'xform': {
                'value': [BRANCH_WIDTH/2, .04, 0, 90, 0, 0]
            }
        },
        {
            'name': "switch_run",
            'type': "DominoRun",
            'args': {
                'extents': DOMINO_LWH,
                'coords': create_x_switch([0, 0], 0, SWITCH_WIDTH, 9),
                'b_mass': DOMINO_MASS
            },
            'xform': {
                'value': [-SWITCH_WIDTH/2, .46, 0, 90, 0, 0]
            }
        },
        {
            'name': "switch_guide1",
            'type': "Box",
            'args': {
                'extents': [DOMINO_LWH[0], DOMINO_LWH[2], DOMINO_LWH[1]],
                'b_mass': DOMINO_MASS
            },
            'parent': "switch_run",
            'xform': {
                'value': [SWITCH_WIDTH-.038, -SWITCH_WIDTH/2-.02,
                          DOMINO_LWH[1]/2,
                          45, 0, 0]
            }
        },
        {
            'name': "switch_guide2",
            'type': "Box",
            'args': {
                'extents': [DOMINO_LWH[0], DOMINO_LWH[2], DOMINO_LWH[1]],
                'b_mass': DOMINO_MASS
            },
            'parent': "switch_run",
            'xform': {
                'value': [SWITCH_WIDTH-.038, -SWITCH_WIDTH/2+.02,
                          DOMINO_LWH[1]/2,
                          -45, 0, 0]
            }
        },
    ],
    'causal_graph': [
        {
            'name': "first_dom_topples",
            'type': "Toppling",
            'args': {
            },
            'children': [
                "left_dom_of_branch_hits_ball",
                "right_dom_of_branch_hits_plank",
            ]
        },
        {
            'name': "left_dom_of_branch_hits_ball",
            'type': "Contact",
            'args': {
            },
            'children': [
                "ball_rolls_on_track",
            ]
        },
        {
            'name': "ball_rolls_on_track",
            'type': "RollingOn",
            'args': {
            },
            'children': [
                "ball_hits_first_dom_of_left_row",
            ]
        },
        {
            'name': "ball_hits_first_dom_of_left_row",
            'type': "Contact",
            'args': {
            },
            'children': [
                "first_dom_of_left_row_topples",
            ]
        },
        {
            'name': "first_dom_of_left_row_topples",
            'type': "Toppling",
            'args': {
            },
            'children': [
                "left_dom_of_switch_topples",
            ]
        },
        {
            'name': "left_dom_of_switch_topples",
            'type': "Toppling",
            'args': {
            },
            'children': [
                "center_dom_of_switch_topples",
            ]
        },
        {
            'name': "right_dom_of_branch_hits_plank",
            'type': "Contact",
            'args': {
            },
            'children': [
                "plank_topples",
            ]
        },
        {
            'name': "plank_topples",
            'type': "Toppling",
            'args': {
            },
            'children': [
                "plank_hits_first_dom_of_wave",
            ]
        },
        {
            'name': "plank_hits_first_dom_of_wave",
            'type': "Contact",
            'args': {
            },
            'children': [
                "first_dom_of_wave_topples",
            ]
        },
        {
            'name': "first_dom_of_wave_topples",
            'type': "Toppling",
            'args': {
            },
            'children': [
                "right_dom_of_switch_topples",
            ]
        },
        {
            'name': "right_dom_of_switch_topples",
            'type': "Toppling",
            'args': {
            },
            'children': [
                "center_dom_of_switch_topples",
            ]
        },
        {
            'name': "center_dom_of_switch_topples",
            'type': "Toppling",
            'args': {
            },
            'children': [
                "{}_end_dom_of_switch_topples".format(FASTER_SIDE),
            ]
        },
        {
            'name': "{}_end_dom_of_switch_topples".format(FASTER_SIDE),
            'type': "Toppling",
            'args': {
            },
            'children': [
                "{}_end_dom_of_switch_remains".format(SLOWER_SIDE),
            ]
        },
        {
            'name': "{}_end_dom_of_switch_remains".format(SLOWER_SIDE),
            'type': "Toppling",
            'args': {
            }
        },
    ]
}
