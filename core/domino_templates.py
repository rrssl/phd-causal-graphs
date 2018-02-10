"""
Domino templates.

"""
import numpy as np


def _linear_transform_2D(coords, position, angle):
    if angle:
        cos = np.cos(angle)
        sin = np.sin(angle)
        rot_t = np.array([[cos, sin], [-sin, cos]])
        coords[:, :2] = coords[:, :2].dot(rot_t)
    coords[:, :2] += position


def create_branch(origin, angle, half_length, half_width, density):
    n_doms = max(2, int(density * half_length))
    coords = np.zeros((3*n_doms-2, 3))
    coords[:n_doms, 0] = np.linspace(0, half_length, n_doms)
    coords[n_doms:2*n_doms-1, 0] = np.linspace(
            coords[n_doms-1, 0], coords[n_doms-1, 0]+half_length, n_doms)[1:]
    coords[n_doms:2*n_doms-1, 1] = np.linspace(0, half_width, n_doms)[1:]
    coords[2*n_doms-1:, 0] = coords[n_doms:2*n_doms-1, 0]
    coords[2*n_doms-1:, 1] = np.linspace(0, -half_width, n_doms)[1:]
    coords[:, 2] = angle
    _linear_transform_2D(coords, origin, np.radians(angle))
    return coords


def create_line(origin, angle, length, density):
    n_doms = max(2, int(density * length))
    coords = np.zeros((n_doms, 3))
    coords[:, 0] = np.linspace(0, length, n_doms)
    coords[:, 2] = angle
    _linear_transform_2D(coords, origin, np.radians(angle))
    return coords


def create_circular_arc(origin, radius, angle_start, angle_stop, density):
    arc_length = radius * np.radians(abs(angle_stop - angle_start))
    n_doms = max(2, int(density * arc_length))
    coords = np.zeros((n_doms, 3))
    angles = np.linspace(angle_start, angle_stop, n_doms)
    angles_rad = np.radians(angles)
    coords[:, 2] = angles + 90
    coords[:, 0] = radius * np.cos(angles_rad)
    coords[:, 1] = radius * np.sin(angles_rad)
    _linear_transform_2D(coords, origin, 0)
    return coords
